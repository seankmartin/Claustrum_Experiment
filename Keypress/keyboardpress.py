import pygame
import time


def text_to_screen(screen, text, x, y, size=50,
                   color=(000, 000, 000), font_type='Comic Sans MS'):
    try:

        text = str(text)
        font = pygame.font.SysFont(font_type, 12)
        text = font.render(text, True, color)
        screen.blit(text, (x, y))

    except Exception as e:
        print('Font Error, saw it coming')
        raise e


def print_info(screen, keys, key_strs, times):
    for i, (key, e_time, key_str) in enumerate(zip(keys, times, key_strs)):
        print_str = "Time {}, key {}:".format(i + 1, key_str)
        text_to_screen(screen, print_str, 20, 20 * (i + 1))
        print_str = "Elapsed time(s) {:.2f}".format(e_time)
        text_to_screen(screen, print_str, 140, 20 *
                       (i + 1))
    text_to_screen(screen, "Press End to reset", 20, 20 * (len(keys) + 2))


def save_times(times, key_strs, filename="times.csv"):
    with open(filename, "w") as f:
        f.write("Number;Key;Time\n")
        for i, (e_time, key_str) in enumerate(zip(times, key_strs)):
            f.write("{};{};{:.2f}\n".format(
                i + 1, key_str, e_time))


def main():
    background_colour = (255, 255, 255)
    (width, height) = (300, 200)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Keypress Timers')
    screen.fill(background_colour)
    pygame.display.flip()

    running = True
    clock = pygame.time.Clock()

    key_strs = ["Z", "X", "C", "V", "Left", "Down", "Right", "Up"]
    keys = [pygame.K_z, pygame.K_x, pygame.K_c, pygame.K_v,
            pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_UP]
    times = [0 for _ in keys]
    counters = [0 for _ in keys]
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                break
            if event.type == pygame.KEYDOWN:
                for i, key in enumerate(keys):
                    if event.key == key:
                        counters[i] = time.time()
                if event.key == pygame.K_r:
                    filename = "times.csv"
                    print("Resetting times, saving current times to {}".format(
                        filename))
                    save_times(times, key_strs, filename)
                    times = [0 for _ in keys]
                    counters = [0 for _ in keys]
            if event.type == pygame.KEYUP:
                for i, key in enumerate(keys):
                    if event.key == key:  # key 'a'
                        counters[i] = time.time() - counters[i]
                        times[i] += counters[i]
                        counter = str(counters[i])
                        counter = counter[:5]
                        print(
                            "You pressed key {} for {} seconds".format(
                                key_strs[i], counter))
            screen.fill(background_colour)
            print_info(screen, keys, key_strs, times)
            pygame.display.update()

            clock.tick(40)
    save_times(times, key_strs, filename)


if __name__ == "__main__":
    main()
