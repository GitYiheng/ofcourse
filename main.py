from algo.arguments import get_args
from algo.runner import Runner


def main():
    Runner(get_args()).run()


if __name__ == '__main__':
    main()
