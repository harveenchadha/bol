from tqdm import tqdm

from .cer import calculate_cer
from .wer import calculate_wer


def calculate_metrics_for_single_file(txt_path, prediction):
    # ground_truth = read_txt_file(txt_path)
    ground_truth = txt_path
    wer = calculate_wer(ground_truth, prediction)
    cer = calculate_cer(ground_truth, prediction)

    return wer, cer, ground_truth


def calculate_metrics_for_batch(txt_dir, preds):
    num_tokens = []
    num_chars = []
    fwer = []
    fcer = []

    for item in tqdm(preds):
        file_name = item[0].split("/")[-1][:-4]
        txt_path = txt_dir + "/" + file_name + ".txt"
        prediction = item[1]

        wer, cer, ground_truth = calculate_metrics_for_single_file(txt_path, prediction)

        fwer.append(wer)
        fcer.append(cer)
        num_tokens.append(len(ground_truth.split()))
        num_chars.append(len(ground_truth.replace(" ", "")))

    wer = sum(fwer) / sum(num_tokens) * 100
    cer = sum(fcer) / sum(num_chars) * 100
    return wer, cer


def calculate_metrics_for_list(ground_truth, preds):
    num_tokens = []
    num_chars = []
    fwer = []
    fcer = []

    for local_gt, local_pred in tqdm(zip(ground_truth, preds)):
        wer, cer, ground_truth = calculate_metrics_for_single_file(local_gt, local_pred)

        fwer.append(wer)
        fcer.append(cer)
        num_tokens.append(len(ground_truth.split()))
        num_chars.append(len(ground_truth.replace(" ", "")))

    wer = sum(fwer) / sum(num_tokens) * 100
    cer = sum(fcer) / sum(num_chars) * 100
    return wer, cer


def evaluate_metrics(ground_truth, preds, mode):
    if mode == "file":
        wer, cer, _ = calculate_metrics_for_single_file(ground_truth, preds)
    elif mode == "dir":
        wer, cer = calculate_metrics_for_batch(ground_truth, preds)
    elif mode == "list":
        wer, cer = calculate_metrics_for_list(ground_truth, preds)

    return wer, cer
