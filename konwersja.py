import json

def rating_to_label(rating):
    return 1 if rating >= 3 else 0

with open('reviews.json', 'r') as file:
    with open('simplified_reviews.json', 'w') as outfile:
        for line in file:
            review = json.loads(line)
            simplified_review = {
                'text': review['text'],
                'label': rating_to_label(review['rating'])
            }
            json.dump(simplified_review, outfile)
