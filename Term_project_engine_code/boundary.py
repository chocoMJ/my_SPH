def apply_bound(positions, velocities, xlim, ylim, i):
    if positions[i, 0] < xlim[0]:
        velocities[i, 0] *= -0.5
        positions[i, 0] = xlim[0]
    elif positions[i, 0] > xlim[1]:
        velocities[i, 0] *= -0.5
        positions[i, 0] = xlim[1]

    if positions[i, 1] < ylim[0]:
        velocities[i, 1] *= -0.5
        positions[i, 1] = ylim[0]
    elif positions[i, 1] > ylim[1]:
        velocities[i, 1] *= -0.5
        positions[i, 1] = ylim[1]