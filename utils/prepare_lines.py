
def read_conll_docs(f, doc_pattern="# begin doc", delim=None, allow_comments=True, comment_pattern="#"):
    """Read sentences from a conll file.

    :param f: `str` The file to read from.
    :param doc_pattern: `str` The beginning of lines that represent new documents
    :param delim: `str` The token between columns in the file.

    Note:
        If you have a sentence where the first token is `#` it will show up in the
        metadata. If this happens you'll need to update you comments to use a different
        comment pattern, something like `# comment:` I recommend having a space in
        you patten so it can't show up as a conll token

    :returns: `Generator[List[List[List[str]]]]`
        A document which is a list of sentences.
    """
    doc, sentence = [], []
    for line in f:
        line = line.rstrip()
        if line.startswith(doc_pattern):
            if doc:
                if sentence:
                    doc.append(sentence)
                yield doc
                doc, sentence = [], []
            continue
        elif allow_comments and not sentence and line.startswith(comment_pattern):
            continue
        if len(line) == 0:
            if sentence:
                doc.append(sentence)
                sentence = []
            continue
        sentence.append(line.split(delim))
    if doc or sentence:
        if sentence:
            doc.append(sentence)
        yield doc 

if __name__ == "__main__":
    file_loc = "data/FA-Farsi/fa_dev."
    with open(file_loc + "conll") as f:
        contents = f.readlines()
    doc = read_conll_docs(contents)

    lines = []
    for sentences in doc:
        for sent in sentences:
            s = ' '.join([word[0] for word in sent])
            s = s.replace(" ,", ",")
            s = s.replace(" .", ".")
            s = s.replace(" :", ":")
            s = s.replace("( ", "(").replace(" )", ")")   
            lines.append(s)
    with open(file_loc + "txt", "w") as f:
        f.writelines(lines)