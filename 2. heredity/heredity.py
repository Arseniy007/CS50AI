import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people: dict, one_gene: set, two_genes: set, have_trait: set) -> float:
    """
    Compute and return a joint probability.
    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    result_probability = 1

    # Loop over all people
    for name, data in people.items():
        name: str
        data: dict
        # Get person's number of genes
        number_of_genes = get_number_of_genes_by_name(name, one_gene, two_genes)

        # Case where we know persons parents
        if data["mother"] and data["father"]:
            # Calculate gene probability
            gene_probability = calculate_child_gene_probability(
                mother_genes=get_number_of_genes_by_name(data["mother"], one_gene, two_genes), 
                father_genes=get_number_of_genes_by_name(data["father"], one_gene, two_genes), 
                child_number_of_genes=number_of_genes
            )
        # Case we know noting about persons parents
        else:
            # Calculate gene probability
            gene_probability = PROBS["gene"][number_of_genes]

        # Calculate probability that person has a trait
        trait_probability = PROBS["trait"][number_of_genes][name in have_trait]

        # Update join probability
        probability_for_person = gene_probability * trait_probability
        result_probability *= probability_for_person

    return result_probability


def calculate_child_gene_probability(mother_genes: int, father_genes: int, child_number_of_genes: int) -> float:
    """
    Calculate probability that child will have given number
    of genes based on the known data about their parents
    """
    probability_of_mothers_gene = calculate_probability_of_passing_gene_to_child(mother_genes)
    probability_of_fathers_gene = calculate_probability_of_passing_gene_to_child(father_genes)

    match child_number_of_genes:
        case 2:
            # If child has 2 genes, multiply the probabilities of inheriting gene from each parent
            gene_probability = probability_of_mothers_gene * probability_of_fathers_gene
        case 1:
            # If child has 1 gene, add probabilities of inheriting gene from each parent and not from other
            gene_probability = (
                probability_of_mothers_gene * (1 - probability_of_fathers_gene) + 
                probability_of_fathers_gene * (1 - probability_of_mothers_gene)
            )
        case 0:
            # If child has no genes, multiply probabilities of not inheriting gene from each parent
            gene_probability = (
                (1 - probability_of_mothers_gene) * (1 - probability_of_fathers_gene)
            )
    return gene_probability


def calculate_probability_of_passing_gene_to_child(number_of_parent_genes: int) -> float:
    """:)"""
    match number_of_parent_genes:
        case 2:
            # 100% - mutation case
            probability = 1 - PROBS["mutation"]
        case 1:
            # 100% - mutation case divided by 2
            probability = 0.5
        case 0:
            # Mutation - only possible way
            probability = PROBS["mutation"]
    return probability


def get_number_of_genes_by_name(person: str, one_gene: set, two_genes: set) -> int:
    """Get number of gives by given name and sets of data"""
    if person in two_genes:
        number_of_genes = 2
    elif person in one_gene:
        number_of_genes = 1
    else:
        number_of_genes = 0
    return number_of_genes


def update(probabilities: dict, one_gene: set, two_genes: set, have_trait: set, p: float) -> None:
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Loop over every person
    for person in probabilities.keys():
        # Update genes value
        number_of_genes = get_number_of_genes_by_name(person, one_gene, two_genes)
        probabilities[person]["gene"][number_of_genes] += p

        # Update traits value
        probabilities[person]["trait"][person in have_trait] += p

    return None


def normalize(probabilities: dict) -> None:
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Loop over every person and normalize each genes and trait probabilities
    for person in probabilities.keys():
        # Calculate missing alpha for genes
        gene_alpha = (
            1 / (
                probabilities[person]["gene"][2] +
                probabilities[person]["gene"][1] +
                probabilities[person]["gene"][0]
            )
        )
        # Normalize genes distribution (update values)
        for i in range(3):
            probabilities[person]["gene"][i] *= gene_alpha

        # Calculate missing alpha for traits
        trait_alpha = (
            1 / (
                probabilities[person]["trait"][True] +
                probabilities[person]["trait"][False]
            )
        )
        # Normalize trait distribution (update values)
        probabilities[person]["trait"][True] *= trait_alpha
        probabilities[person]["trait"][False] *= trait_alpha

    return None


if __name__ == "__main__":
    main()
