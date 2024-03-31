import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: dict, page: str, damping_factor: float) -> dict:
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    related_pages = corpus[page]
    probability_distribution = dict()

    if related_pages:
        # Calculate probability according to formula
        related_page_probability = damping_factor / len(related_pages)
        random_page_probability = (1 - damping_factor) / len(corpus)
        related_page_probability += random_page_probability

        # Distribute probabilities
        for page in corpus:
            if page in related_pages:
                probability_distribution[page] = related_page_probability
            else:
                probability_distribution[page] = random_page_probability
    else:
        # Set all equal random probability for all pages in corpus
        random_page_probability = 1 / len(corpus)
        probability_distribution = dict.fromkeys(corpus.keys(), random_page_probability)
    return probability_distribution


def sample_pagerank(corpus: dict, damping_factor: float, n: int) -> dict:
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages_occurrence = dict()
    next_sample_pages = dict()

    # Take 10.000 samples
    for sample in range(n):
        if sample == 0:
            # Choose page randomly
            page = random.choice(list(corpus.keys()))
            next_sample_pages = transition_model(corpus, page, damping_factor)
            pages_occurrence[page] = 1
        else:
            # Choose page based on previous sample transition model
            page = random.choices(list(next_sample_pages.keys()), weights=next_sample_pages.values(), k=1)[0]
            next_sample_pages = transition_model(corpus, page, damping_factor)
            if page in pages_occurrence.keys():
                pages_occurrence[page] += 1
            else:
                pages_occurrence[page] = 1

    # Divide each value by n (10.000) in order to get probability instead of number of occurrences
    pages_page_rank = {page:float(pages_occurrence[page]) / n for page in pages_occurrence}
    return pages_page_rank


def iterate_pagerank(corpus: dict, damping_factor: float) -> dict:
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Get reversed corpus dict
    reversed_corpus: dict = reverse_corpus(corpus)

    # Assign each page equal 1 / n rank
    total_num_of_pages = len(corpus)
    starting_rank = 1 / total_num_of_pages
    pages_page_rank = {page:starting_rank for page in corpus}

    # Calculate first part of Pagerank formula
    random_page_probability = (1 - damping_factor) / total_num_of_pages

    # Infinite loop
    while True:
        # Set flag
        stop_iterating = True

        for page in corpus.keys():
            # Get all pages linking to current page
            source_pages: set = reversed_corpus[page]

            # Sum info about pages linking to current page
            sum_of_source_pages_ranks = 0
            for source_page in source_pages:
                source_page_rank = pages_page_rank[source_page] 
                num_of_links = len(corpus[source_page])
                sum_of_source_pages_ranks += source_page_rank / num_of_links
            
            # Calculate new page rank
            old_pagerank = pages_page_rank[page]
            new_pagerank = random_page_probability + damping_factor * sum_of_source_pages_ranks
            pages_page_rank[page] = new_pagerank

            # Calculate difference between old and new ranks
            ranks_difference = abs(new_pagerank - old_pagerank)
            print(ranks_difference)
            if ranks_difference > 0.001:
                stop_iterating = False

        # Check if we need to continue iterating        
        if stop_iterating:
            return pages_page_rank
        

def reverse_corpus(corpus: dict) -> dict:
    """
    Create dict out of corpus where keys will be the titles of the pages
    and values: the pages that link to key-page rather than pages to which key-page links
    """
    reversed_corpus = dict()

    # Add link to every page if page has no links
    for page in corpus.keys():
        if not corpus[page]:
            corpus[page] = corpus.keys()

    for page in corpus.keys():
        # Find all pages linking to current page
        source_pages = set()
        for source_page, target_pages in corpus.items():
            if page in target_pages:
                source_pages.add(source_page)
        reversed_corpus[page] = source_pages
    return reversed_corpus


if __name__ == "__main__":
    main()
