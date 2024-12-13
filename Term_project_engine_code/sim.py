import pygame
import numpy as np
from neighbour_search import *
from update import *
from compute_force import *
from compute_densities_pressure import *

def draw_text(surface, text, position, font, color=(0, 0, 0)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

pygame.init()   

WIDTH = 600
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fluid Particle Simulation - Settings")
BACKGROUND_COLOR = (255, 255, 255)  # 하얀색
TEXT_COLOR = (0, 0, 0)  # 검정
BUTTON_COLOR = (0, 200, 0)  # 초록
BUTTON_TEXT_COLOR = (255, 255, 255)  # 하얀색
ACTIVE_INPUT_COLOR = (200, 200, 200)  # gray -> 현재 선택한 버튼 표시

font = pygame.font.Font(None, 36)
input_font = pygame.font.Font(None, 28)

parameters = {
    "mass": 1.0,
    "k": 20.0,
    "rest_density": 1.0,
    "visc": 1.0,
    "smoothing_length": 5.0
}

def draw_input_box(label, value, position, input_rects, active_input):
    label_surface = font.render(label, True, TEXT_COLOR)
    screen.blit(label_surface, (position[0], position[1]))

    input_rect = pygame.Rect(position[0] + 300, position[1], 150, 30)
    box_color = ACTIVE_INPUT_COLOR if active_input == label else BACKGROUND_COLOR
    pygame.draw.rect(screen, box_color, input_rect)
    pygame.draw.rect(screen, TEXT_COLOR, input_rect, 2)

    input_text = input_font.render(value, True, TEXT_COLOR)
    screen.blit(input_text, (input_rect.x + 5, input_rect.y + 5))

    input_rects[label] = input_rect

# GUI loop
running = True
start_simulation = False
input_rects = {}
input_values = {key: str(value) for key, value in parameters.items()}
error_message = ""
active_input = None

while running and not start_simulation:
    screen.fill(BACKGROUND_COLOR)

    y_offset = 50
    for param, value in input_values.items():
        draw_input_box(param, value, (50, y_offset), input_rects, active_input)
        y_offset += 50

    button_rect = pygame.Rect(250, y_offset + 30, 100, 40)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    draw_text(screen, "Start", (button_rect.x + 20, button_rect.y + 5), font, BUTTON_TEXT_COLOR)

    if error_message:
        draw_text(screen, error_message, (50, y_offset + 80), font, (255, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                try:
                    for param in parameters:
                        parameters[param] = float(input_values[param])
                    start_simulation = True
                except ValueError:
                    error_message = "All values must be numbers!"
            else:
                for label, rect in input_rects.items():
                    if rect.collidepoint(event.pos):
                        active_input = label
                        break
                else:
                    active_input = None

        if event.type == pygame.KEYDOWN and active_input:
            if event.key == pygame.K_RETURN:
                active_input = None
            elif event.key == pygame.K_BACKSPACE:
                input_values[active_input] = input_values[active_input][:-1]
            else:
                input_values[active_input] += event.unicode

    pygame.display.flip()

if start_simulation:
    pygame.display.set_caption("Fluid Particle Simulation")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    SMOOTHING_LENGTH = int(parameters["smoothing_length"])
    xlim = np.array([SMOOTHING_LENGTH, WIDTH - SMOOTHING_LENGTH])
    ylim = np.array([SMOOTHING_LENGTH, HEIGHT - SMOOTHING_LENGTH])

    mass = parameters["mass"]
    k = parameters["k"]
    rest_density = parameters["rest_density"]
    visc = parameters["visc"]
    gravity = mass * np.array([0, 0.1])
    time_step = 0.1
    cells, x_cells, y_cells = create_spatial_hash(WIDTH, HEIGHT, SMOOTHING_LENGTH)

    dam_ylim = (3 * ylim[1] // 4, 7* ylim[1] // 8)
    dam_xlim = (int((xlim[1] - xlim[0]) * 0.05), int((xlim[1] - xlim[0]) * 0.95))

    positions = []
    num_particles = 0

    for y in range(*dam_ylim, int(SMOOTHING_LENGTH)):
        for x in range(*dam_xlim, int(SMOOTHING_LENGTH)):
            pos_x = x + np.random.uniform(0, SMOOTHING_LENGTH * 0.1)
            pos_y = y + np.random.uniform(0, SMOOTHING_LENGTH * 0.1)
            positions.append([pos_x, pos_y])
            add_to_cell(cells, x_cells, y_cells, [pos_x, pos_y], num_particles, SMOOTHING_LENGTH)
            num_particles += 1

    positions = np.array(positions)
    velocities = np.zeros((num_particles, 2))
    forces = np.zeros((num_particles, 2))
    densities = np.zeros(num_particles)
    pressures = np.zeros(num_particles)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BACKGROUND_COLOR)
        compute_density_pressure(positions, densities, pressures, SMOOTHING_LENGTH, mass, k, rest_density, cells, x_cells, y_cells)
        compute_force(positions, velocities, pressures, densities, SMOOTHING_LENGTH, mass, visc, gravity, forces, cells, x_cells, y_cells)
        update(positions, velocities, forces, densities, time_step, xlim, ylim, cells, x_cells, y_cells, SMOOTHING_LENGTH)

        for pos in positions:
            pygame.draw.circle(screen, (0, 0, 255), (int(pos[0]), int(pos[1])), SMOOTHING_LENGTH)

        pygame.display.flip()
        clock.tick(60)

pygame.quit()
