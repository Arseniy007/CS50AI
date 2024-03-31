import nltk
from nltk.tokenize import word_tokenize
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | VP | NP VP Conj VP | NP VP Conj NP VP
NP -> N | Det N | Det NP | N NP | Conj NP | Adv NP | Adj NP | P NP 
VP -> V | V NP | VP NP | Adv VP | VP Adv | V Adv | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence: str) -> list:
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = word_tokenize(sentence)
    result = list()
    for word in words:
        for char in word:
            if char.isalpha():
                result.append(word.lower())
                break
    return result


def np_chunk(tree: nltk.tree) -> list:
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    result = list()
    parented_tree = nltk.tree.ParentedTree.convert(tree)
    subtrees = parented_tree.subtrees()
    for subtree in subtrees:
        if subtree.label() == "N":
            possible_np_chunk = subtree.parent()
            flag = True
            for other_subtree in possible_np_chunk.subtrees():
                if other_subtree.label() == "NP":
                    flag = False
            if flag:
                result.append(possible_np_chunk)
    return result


if __name__ == "__main__":
    main()
