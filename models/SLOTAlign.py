import torch
import dgl
from dgl.nn.pytorch import GINConv, GraphConv
import numpy as np
import random
import copy
from torch_geometric.utils import to_dense_adj


class SLOTAlign:
    def __init__(self, config):
        self.joint_epoch = config['joint_epoch']
        self.truncate = config['truncate']
        self.feat_noise = config['feat_noise']
        self.noise_type = config['noise_type']
        self.bases = config['bases']
        self.step_size = config['step_size']
        self.gw_beta = config['gw_beta']

    def run(self, G1, G2, alignment):
        Aadj = to_dense_adj(G1.edge_index).squeeze(0).cpu().numpy()
        Badj = to_dense_adj(G2.edge_index).squeeze(0).cpu().numpy()
        Afeat = np.ones((G1.num_nodes, 1), dtype=np.float32)
        Bfeat = np.ones((G2.num_nodes, 1), dtype=np.float32)
        ground_truth = alignment

        Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
        Ag = dgl.graph(np.nonzero(Aadj), num_nodes=Adim)
        Bg = dgl.graph(np.nonzero(Badj), num_nodes=Bdim)
        Afeat -= Afeat.mean(0)
        Bfeat -= Bfeat.mean(0)

        if self.truncate:
            Afeat = Afeat[:, :100]
            Bfeat = Bfeat[:, :100]

        if self.noise_type == 1:
            Bfeat = self.feature_permutation(Bfeat, ratio=self.feat_noise)
        elif self.noise_type == 2:
            Bfeat = self.feature_truncation(Bfeat, ratio=self.feat_noise)
        elif self.noise_type == 3:
            Bfeat = self.feature_compression(Bfeat, ratio=self.feat_noise)

        print('feature size:', Afeat.shape, Bfeat.shape)

        Afeat = torch.tensor(Afeat).float().cuda()
        Bfeat = torch.tensor(Bfeat).float().cuda()

        layers = self.bases - 2
        conv = GraphConv(0, 0, norm='both', weight=False, bias=False)
        Afeats = [torch.tensor(Afeat)]
        Bfeats = [torch.tensor(Bfeat)]
        Ag = Ag.to('cuda:0')
        Bg = Bg.to('cuda:0')
        for i in range(layers):
            Afeats.append(conv(dgl.add_self_loop(Ag), torch.tensor(Afeats[-1])).detach().clone())
            Bfeats.append(conv(dgl.add_self_loop(Bg), torch.tensor(Bfeats[-1])).detach().clone())

        Asims, Bsims = [Ag.adj().to_dense().cuda()], [Bg.adj().to_dense().cuda()]
        for i in range(len(Afeats)):
            Afeat = Afeats[i]
            Bfeat = Bfeats[i]
            Afeat = Afeat / (Afeat.norm(dim=1)[:, None] + 1e-16)
            Bfeat = Bfeat / (Bfeat.norm(dim=1)[:, None] + 1e-16)
            Asim = Afeat.mm(Afeat.T)
            Bsim = Bfeat.mm(Bfeat.T)
            Asims.append(Asim)
            Bsims.append(Bsim)

        Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
        a = torch.ones([Adim, 1]).cuda() / Adim
        b = torch.ones([Bdim, 1]).cuda() / Bdim
        X = a @ b.T
        As = torch.stack(Asims, dim=2)
        Bs = torch.stack(Bsims, dim=2)

        alpha0 = np.ones(layers + 2).astype(np.float32) / (layers + 2)
        beta0 = np.ones(layers + 2).astype(np.float32) / (layers + 2)
        for ii in range(self.joint_epoch):
            alpha = torch.autograd.Variable(torch.tensor(alpha0)).cuda()
            alpha.requires_grad = True
            beta = torch.autograd.Variable(torch.tensor(beta0)).cuda()
            beta.requires_grad = True
            print(As.dtype, Bs.dtype, alpha.dtype, beta.dtype)
            A = (As * alpha).sum(2)
            B = (Bs * beta).sum(2)
            print(X.type(), A.type(), B.type())
            objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ X @ B @ X.T)
            alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
            alpha = alpha - self.step_size * alpha_grad
            alpha0 = alpha.detach().cpu().numpy()
            alpha0 = self.euclidean_proj_simplex(alpha0)
            beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
            beta = beta - self.step_size * beta_grad
            beta0 = beta.detach().cpu().numpy()
            beta0 = self.euclidean_proj_simplex(beta0)
            X = self.gw_torch(A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=self.gw_beta,
                         outer_iter=1, inner_iter=50).clone().detach()
            if ii == self.joint_epoch - 1:
                print(alpha0, beta0)
                X = self.gw_torch(A.clone().detach(), B.clone().detach(), a, b, X, beta=self.gw_beta,
                             outer_iter=self.epoch - self.joint_epoch, inner_iter=20, gt=ground_truth)

    def gw_torch(self, cost_s, cost_t, p_s=None, p_t=None, trans0=None, beta=1e-1, error_bound=1e-10,
                 outer_iter=200, inner_iter=1, gt=None):
        # a = torch.ones_like(p_s)/p_s.shape[0]
        if trans0 is None:
            trans0 = p_s @ p_t.T
        for oi in range(outer_iter):
            a = torch.ones_like(p_s) / p_s.shape[0]
            cost = - 2 * (cost_s @ trans0 @ cost_t.T)
            kernel = torch.exp(-cost / beta) * trans0
            for ii in range(inner_iter):
                b = p_t / (kernel.T @ a)
                a_new = p_s / (kernel @ b)
                relative_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
                a = a_new
                if relative_error < error_bound:
                    break
            trans = (a @ b.T) * kernel
            relative_error = torch.sum(torch.abs(trans - trans0)) / torch.sum(torch.abs(trans0))
            if relative_error < error_bound:
                print(relative_error)
                break
            trans0 = trans
            if oi % 20 == 0 and oi > 2:
                if gt is not None:
                    res = trans0.T.cpu().numpy()
                    a1, a5, a10 = self.my_check_align1(res, gt)
                print(oi, (cost_s ** 2).mean() + (cost_t ** 2).mean() - torch.trace(cost_s @ trans @ cost_t @ trans.T),
                      a1, a5, a10)
        return trans

    def euclidean_proj_simplex(self, v, s=1):
        """ Compute the Euclidean projection on a positive simplex
        Solves the optimisation problem (using the algorithm from [1]):
            min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
        Parameters
        ----------
        v: (n,) numpy array,
           n-dimensional vector to project
        s: int, optional, default: 1,
           radius of the simplex
        Returns
        -------
        w: (n,) numpy array,
           Euclidean projection of v on the simplex
        Notes
        -----
        The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
        Better alternatives exist for high-dimensional sparse vectors (cf. [1])
        However, this implementation still easily scales to millions of dimensions.
        References
        ----------
        [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
            International Conference on Machine Learning (ICML 2008)
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
        """
        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (v - theta).clip(min=0)
        return w

    def feature_permutation(self, Bfeat, ratio=0.):
        feat_dim = Bfeat.shape[1]
        permutation_featdim = int(feat_dim * ratio + 0.01)
        permutation_ids = random.sample(range(feat_dim), permutation_featdim)
        permutation_ids2 = copy.deepcopy(permutation_ids)
        random.shuffle(permutation_ids2)
        Bfeat[:, permutation_ids] = Bfeat[:, permutation_ids2]
        return Bfeat

    def my_check_align1(self, pred, ground_truth, result_file=None):
        g_map = {}
        for i in range(ground_truth.size(1)):
            g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
        g_list = list(g_map.keys())
        ind = (-pred).argsort(axis=1)[:, :10]
        a1, a5, a10 = 0, 0, 0
        for i, node in enumerate(g_list):
            for j in range(10):
                if ind[node, j].item() == g_map[node]:
                    if j < 1:
                        a1 += 1
                    if j < 5:
                        a5 += 1
                    if j < 10:
                        a10 += 1
        a1 /= len(g_list)
        a5 /= len(g_list)
        a10 /= len(g_list)
        print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%%' % (a1 * 100, a5 * 100, a10*100))
        return a1,a5,a10
