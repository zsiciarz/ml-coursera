from __future__ import print_function

import re


def process_email(email_contents):
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('\d+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    return []


if __name__ == '__main__':
    file_contents = open('../../octave/mlclass-ex6/emailSample1.txt', 'r').read()
    word_indices = process_email(file_contents)
    print('Word indices:\n%s' % word_indices)
