
def test_reflections_1() -> None:
    """
    Compose does not work properly for chained reflection and rotation.
    """

    pts_local = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [2, 2], [1, 2]])

    plt.scatter(pts_local[:, 0], pts_local[:, 1], 10, color="r", marker=".")
    plt.plot(pts_local[:, 0], pts_local[:, 1], color="r")

    R = rotmat2d(45)  # 45 degree rotation
    t = np.array([1, 1])
    s = 2.0
    world_Sim2_local = Sim2(R, t, s)

    pts_world = world_Sim2_local.transform_from(pts_local)

    plt.scatter(pts_world[:, 0], pts_world[:, 1], 10, color="b", marker=".")
    plt.plot(pts_world[:, 0], pts_world[:, 1], color="b")

    plt.scatter(pts_world[:, 0], pts_world[:, 1], 10, color="b", marker=".")
    plt.plot(pts_world[:, 0], pts_world[:, 1], color="b")

    R_refl = np.array([[-1.0, 0], [0, 1]])
    reflectedworld_Sim2_world = Sim2(R_refl, t=np.zeros(2), s=1.0)

    # import pdb; pdb.set_trace()
    pts_reflworld = reflectedworld_Sim2_world.transform_from(pts_world)

    plt.scatter(pts_reflworld[:, 0], pts_reflworld[:, 1], 100, color="g", marker=".")
    plt.plot(pts_reflworld[:, 0], pts_reflworld[:, 1], color="g", alpha=0.3)

    plt.scatter(-pts_world[:, 0], pts_world[:, 1], 10, color="m", marker=".")
    plt.plot(-pts_world[:, 0], pts_world[:, 1], color="m", alpha=0.3)

    plt.axis("equal")
    plt.show()


def test_reflections_2() -> None:
    """
    Try reflection -> rotation -> compare relative poses

    rotation -> reflection -> compare relative poses

    Relative pose is identical, but configuration will be different in the absolute world frame
    """
    pts_local = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [2, 2], [1, 2]])

    R_refl = np.array([[-1.0, 0], [0, 1]])
    identity_Sim2_reflected = Sim2(R_refl, t=np.zeros(2), s=1.0)

    # pts_refl = identity_Sim2_reflected.transform_from(pts_local)
    pts_refl = pts_local

    R = rotmat2d(45)  # 45 degree rotation
    t = np.array([1, 1])
    s = 1.0
    world_Sim2_i1 = Sim2(R, t, s)

    R = rotmat2d(45)  # 45 degree rotation
    t = np.array([1, 2])
    s = 1.0
    world_Sim2_i2 = Sim2(R, t, s)

    pts_i1 = world_Sim2_i1.transform_from(pts_refl)
    pts_i2 = world_Sim2_i2.transform_from(pts_refl)

    pts_i1 = identity_Sim2_reflected.transform_from(pts_i1)
    pts_i2 = identity_Sim2_reflected.transform_from(pts_i2)

    plt.scatter(pts_i1[:, 0], pts_i1[:, 1], 10, color="b", marker=".")
    plt.plot(pts_i1[:, 0], pts_i1[:, 1], color="b")

    plt.scatter(pts_i2[:, 0], pts_i2[:, 1], 10, color="g", marker=".")
    plt.plot(pts_i2[:, 0], pts_i2[:, 1], color="g")

    plt.axis("equal")
    plt.show()