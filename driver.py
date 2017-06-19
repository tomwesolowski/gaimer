import pacman.maps


def main():
    env = pacman.maps.get_very_simple_environment()
    env.run(keep=True)

if __name__ == '__main__':
    main()
