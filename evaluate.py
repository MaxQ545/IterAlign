class Evaluator:
    """
    Evaluator class for calculating Hits@1, Hits@5, and MRR metrics.
    """
    def __init__(self, align_links, align_ranks, alignment, time):
        """
        Initializes the evaluator.

        Args:
            align_links (list of list): [aligned nodes from graph1, aligned nodes from graph2].
            align_ranks (list of list): Ranked candidate nodes for each node in graph1.
            alignment (dict): Ground-truth alignment between graph1 and graph2.
        """
        self.align_links = align_links
        self.align_ranks = align_ranks
        self.alignment = alignment
        self.time = time

    def evaluate(self, first_select_numbers):
        """
        Evaluates Hits@1, Hits@5, MRR, and CENA Accuracy.
        """
        top1_count = 0
        top5_count = 0
        MRR_total = 0.0
        total = len(self.align_links[0])
        wrong_idx = []
        first_correct = 0

        for idx in range(total):
            node1 = self.align_links[0][idx]
            node2 = self.align_links[1][idx]
            node1_str = str(node1)
            if node1_str not in self.alignment.keys():
                continue
            true_node2 = self.alignment[node1_str]

            if node2 == true_node2:  # Top-1 (accuracy)
                top1_count += 1
                if idx < first_select_numbers:
                    first_correct += 1
            else:
                wrong_idx.append(idx)

            if len(self.align_ranks) > 0:
                candidate_nodes = self.align_ranks[idx]
                if true_node2 in candidate_nodes[:5]:  # Top-5
                    top5_count += 1
                if true_node2 in candidate_nodes:  # MRR
                    rank = candidate_nodes.index(true_node2) + 1
                    MRR_total += 1.0 / rank


        # Calculate metrics
        top1_accuracy = top1_count / total
        top5_accuracy = top5_count / total
        MRR = MRR_total / total

        print(f"\nEvaluations")
        print(f"{'=' * 50}")
        print(f"Number of Alignments: {total}")
        print(f"Acc His@1: {top1_accuracy:.4f}")
        print(f"Acc His@5: {top5_accuracy:.4f}")
        print(f"MRR: {MRR:.4f}")
        print("Wrong alignment links:", wrong_idx[:100])
        print("Running time(s):", self.time)

        if first_select_numbers > 0:
            print("First select accuracy:", first_correct / first_select_numbers)
