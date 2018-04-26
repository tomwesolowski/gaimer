from pacman import maps

def main():
    env = maps.get_simple_environment()
    env.run(keep=True)

if __name__ == '__main__':
    main()
