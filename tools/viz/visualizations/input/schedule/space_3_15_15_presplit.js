////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    const common = GetCommonConstants();

    return {
        name: "Schedule",
        pixel_width: common.pixel_width,
        pixel_height: common.pixel_height,
        background_color: common.background_color,
        ambient_light_color: common.ambient_light_color,
        ambient_light_intensity: common.ambient_light_intensity,
        shadow_map_resolution: common.shadow_map_resolution
    }
}

async function RunViz(Viz, SceneView) {
    const common = GetCommonConstants();

    let d0 = 3;
    let d1 = 15;
    let d2 = 15;

    const space1 = Viz.CreateIterationSpace({
        shape: [d0, d1, d2],
        size: 1.0,
        padding: 0.15,
        position: { x: 0, y: 0, z: 0 },
        color: common.ball_color,
        rotation: { x: 0.4, y: 2.65, z: 3.1 }
    });

    const axis1 = Viz.CreateAxisLabel({
        space: space1,
        axis0: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5,
            arrowhead_length: 1,
            arrow_length: d0 - 0.5,
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "i",
            label_pos: 0.5,
            label_size: 1
        },
        axis1: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_thickness: 0.5,
            arrowhead_length: 1,
            arrow_length: 15,
            label: "j",
            label_pos: 0.3,
            label_size: 1,
            arrow_start_offset: 0,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.FRONT,
        },
        axis2: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_thickness: 0.5,
            arrowhead_length: 2,
            arrow_length: d2,
            arrow_start_offset: -1,
            label: "k",
            label_pos: 0.3,
            label_size: 1,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    });


    const lighting1 = Viz.CreateIterationSpaceLighting({
        space: space1,
        show_lights: false,
        top_light: { color: 0xffffff, panel_distance: 20, intensity: 0.6, distance: 35, decay: 2, light_count: { x: 2, y: 2 }, light_stride: { x: 12, y: 16 }, offset: { x: 0, y: -5 } },
        left_light: { color: 0xffffff, panel_distance: 12, intensity: 1.2, distance: 30, decay: 2, light_count: { x: 2, y: 1 }, light_stride: { x: 12, y: 10 }, offset: { x: 0, y: 8 } },
        right_light: { color: 0xffffff, panel_distance: 20, intensity: 1, distance: 40, decay: 2.2, light_count: { x: 2, y: 1 }, light_stride: { x: 18, y: 18 }, offset: { x: 3, y: 8 } },
    });

    for (let i = 0; i < d0; ++i) {
        for (let j = 12; j < d1; ++j) {
            for (let k = 0; k < d2; ++k) {
                space1.set_child_color(
                    [i, j, k],
                    common.padding_color1
                );
            }
        }
    }

    Viz.camera.position.z = 45;
    Viz.camera.position.y = -3;
    Viz.camera.position.x = -1;
    Viz.camera.set_fov_zoom(0.6);

    await Viz.SaveImage("space_3_15_15_presplit");
}

