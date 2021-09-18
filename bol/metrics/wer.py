import Levenshtein as Lev


def wer_single(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance("".join(w1), "".join(w2))


def calculate_wer(source, target):
    wer_local = ""
    try:
        wer_local = wer_single(source, target)
    except:
        print(source)
        return len(source.split(" "))
    return wer_local


def wer_for_evaluate(ground_truth, predictions):
    num_tokens = []
    fwer = []

    dict_gt = {}

    for item in ground_truth:
        gt_file_name = item["text_file_name"].split("/")[-1].split(".")[0]
        dict_gt[gt_file_name] = item

    for pred in predictions:
        pred_file_name = pred["file"].split("/")[-1].split(".")[0]
        gt = dict_gt[pred_file_name]

        fwer.append(wer_single(gt["text_file_content"], pred["transcription"]))
        num_tokens.append(len(gt["text_file_content"].split()))

    wer = sum(fwer) / sum(num_tokens) * 100
    return wer


def wer(ground_truth, predictions):
    num_tokens = []
    fwer = []
    for gt, pred in zip(ground_truth, predictions):
        fwer.append(wer_single(gt, pred))
        num_tokens.append(len(gt.split()))

    wer = sum(fwer) / sum(num_tokens) * 100

    return wer
