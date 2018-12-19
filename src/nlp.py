import re
import string


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def remove_non_letters(review):
    words = ''.join([x + '' for x in review if x in string.ascii_letters + ' '])
    return re.sub('  ', ' ', words)


def preprocess_phrase(reviews):

    reviews = [REPLACE_NO_SPACE.sub("", str(line).lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    reviews = [remove_non_letters(r) for r in reviews]

    return reviews
