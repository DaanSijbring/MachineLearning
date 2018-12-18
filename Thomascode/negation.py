def negate_sequence(words):
    prev = []
    result = []
    neg = ["not", "n't", "no"]
    j = 0
    for word in words:
        if word not in neg:
            if prev in neg:
                result.append('not_' + word)
            else:
                result.append(word)
            j += 1
        prev = word
    return(result)