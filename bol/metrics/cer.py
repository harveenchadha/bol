import Levenshtein as Lev


def cer_single(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(
        " ", ""
    ), s2.replace(" ", "")
    return Lev.distance(s1, s2)


def calculate_cer(source, target):
    try:
        cer_local = cer_single(source, target)
    except:
        return len(source.replace(" ", ""))

    return cer_local


def cer_for_evaluate(ground_truth, predictions):
    num_tokens = []
    fcer = []

    dict_gt = {}

    for item in ground_truth:
        gt_file_name = item["text_file_name"].split("/")[-1].split(".")[0]
        dict_gt[gt_file_name] = item

    for pred in predictions:
        pred_file_name = pred["file"].split("/")[-1].split(".")[0]
        gt = dict_gt[pred_file_name]
        fcer.append(cer_single(gt["text_file_content"], pred["transcription"]))
        num_tokens.append(len(gt["text_file_content"].replace(" ", "")))

    cer = sum(fcer) / sum(num_tokens) * 100
    return cer


def cer(ground_truth, predictions):
    num_chars = []
    fcer = []
    for gt, pred in zip(ground_truth, predictions):
        fcer.append(cer_single(gt, pred))
        num_chars.append(len(gt.replace(" ", "")))

    cer = sum(fcer) / sum(num_chars) * 100

    return cer
