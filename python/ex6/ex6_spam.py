from __future__ import print_function


def process_email(email_contents):
    return []


if __name__ == '__main__':
    file_contents = open('../../octave/mlclass-ex6/emailSample1.txt', 'r').read()
    word_indices = process_email(file_contents)
    print('Word indices:\n%s' % word_indices)
