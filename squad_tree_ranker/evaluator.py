from utils import f1

NA_THRESHOLD = 0.28836128799999994
HAS_ANSWER_THRESHOLD = 1.0 - NA_THRESHOLD


class Evaluator:
    def __init__(self, category='train'):
        self.category = category
        self.has_answer_count, self.no_answer_count = 0., 0.

        self.batches = 0

        self.has_answer_loss = 0.
        self.global_has_answer_loss = 0.
        self.span_loss = 0.

        self.span_recalls = 0.
        self.span_precisions = 0.
        self.span_ems = 0.
        self.span_f1s = 0.

        self.has_answer_true_positives = 0.
        self.has_answer_predicted_positives = 0.

        self.global_has_answer_true_positives = 0.
        self.global_has_answer_predicted_positives = 0.

        self.correct_has_answer = 0.
        self.global_correct_has_answer = 0.
        self.system_f1 = 0.

    def add_predictions(self, batch=None, span_confidence=None, span_loss=None, answer_loss=None,
                        predicted_node_idxs=None, answer_proba=None, has_answer=None, global_answer_proba=None,
                        global_answer_loss=None):
        self.batches += 1

        batch_size = len(has_answer)
        answerable = sum(has_answer)
        self.has_answer_count += answerable
        self.no_answer_count += batch_size - answerable

        self.span_loss += span_loss
        self.has_answer_loss += answer_loss
        self.global_has_answer_loss += global_answer_loss

        em, recall, precision, f1, allf1s = batch.precision_recall(predicted_node_idxs)
        self.span_ems += sum(em)
        self.span_recalls += sum(recall)
        self.span_precisions += sum(precision)
        self.span_f1s += sum(f1)

        for j, proba in enumerate(answer_proba):
            true_has_answer = has_answer[j] == 1
            predicted_answer = proba >= 0.5

            if true_has_answer == predicted_answer:
                self.correct_has_answer += 1

            if predicted_answer:
                self.has_answer_predicted_positives += 1
                if true_has_answer:
                    self.has_answer_true_positives += 1

        for j, proba in enumerate(global_answer_proba):
            true_has_answer = has_answer[j] == 1
            predicted_answer = proba >= 0.5

            if true_has_answer == predicted_answer:
                self.global_correct_has_answer += 1

            if predicted_answer:
                self.global_has_answer_predicted_positives += 1
                if true_has_answer:
                    self.global_has_answer_true_positives += 1

        for batch_idx, (p1, p2, p3) in enumerate(zip(global_answer_proba, answer_proba, span_confidence)):
            true_has_answer = has_answer[batch_idx] == 1
            predicted_answer = (0.4 * p1 + 0.3 * p2 + 0.3 * p3) >= HAS_ANSWER_THRESHOLD
            if true_has_answer == predicted_answer:
                if true_has_answer:
                    self.system_f1 += allf1s[batch_idx]
                else:
                    self.system_f1 += 1.0

    def span_f1(self):
        return self.span_f1s / float(self.has_answer_count)

    def avg_span_recall(self):
        return self.span_recalls / float(self.has_answer_count)

    def avg_span_precision(self):
        return self.span_precisions / float(self.has_answer_count)

    def avg_span_em(self):
        return self.span_ems / float(self.has_answer_count)

    def answer_f1(self):
        return f1(self.avg_answer_recall(), self.avg_answer_precision())

    def global_answer_f1(self):
        return f1(self.global_avg_answer_recall(), self.global_avg_answer_precision())

    def avg_answer_recall(self):
        return self.has_answer_true_positives / float(self.has_answer_count)

    def global_avg_answer_recall(self):
        return self.global_has_answer_true_positives / float(self.has_answer_count)

    def avg_answer_precision(self):
        return 0.0 if self.has_answer_predicted_positives == 0.0 else (
            self.has_answer_true_positives / float(self.has_answer_predicted_positives))

    def global_avg_answer_precision(self):
        return 0.0 if self.global_has_answer_predicted_positives == 0.0 else (
            self.global_has_answer_true_positives / float(self.global_has_answer_predicted_positives))

    def avg_answer_accuracy(self):
        return self.correct_has_answer / float(self.has_answer_count + self.no_answer_count)

    def global_avg_answer_accuracy(self):
        return self.global_correct_has_answer / float(self.has_answer_count + self.no_answer_count)

    def avg_span_loss(self):
        return self.span_loss / float(self.batches)

    def avg_answer_loss(self):
        return self.has_answer_loss / float(self.has_answer_count + self.no_answer_count)

    def global_avg_answer_loss(self):
        return self.global_has_answer_loss / float(self.has_answer_count + self.no_answer_count)

    def avg_system_f1(self):
        return self.system_f1 / float(self.has_answer_count + self.no_answer_count)

    def eval_string(self):
        title_string = 'Train' if self.category == 'train' else 'Dev'
        return '%s: System F1=%.3f, Span Loss=%.3f, F1=%.3f, R=%.3f, P=%.3f, EM=%.2f. Global-Has-Answer Loss=%.4f, ' \
               'Acc.=%.2f, R=%.3f, P=%.3f. Has-Answer Loss=%.4f Acc.=%.2f, R=%.3f, P=%.3f.' %\
               (title_string, self.avg_system_f1(), self.avg_span_loss(), self.span_f1(), self.avg_span_recall(),
                self.avg_span_precision(), self.avg_span_em(), self.global_avg_answer_loss(),
                self.global_avg_answer_accuracy(), self.global_avg_answer_recall(), self.global_avg_answer_precision(),
                self.avg_answer_loss(), self.avg_answer_accuracy(),
                self.avg_answer_recall(), self.avg_answer_precision())
