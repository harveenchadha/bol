import Levenshtein as Lev

def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)


def calculate_cer(source, target):
    try:
        cer_local = cer(source, target)
    except:
        return len(source.replace(' ',''))
        
    return cer_local