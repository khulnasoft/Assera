////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    const common = GetCommonConstants();
    
    return {
        name: "Fuse",
        pixel_width: common.pixel_width,
        pixel_height: common.pixel_height,
        background_color: common.background_color,
        ambient_light_color: common.ambient_light_color,
        ambient_light_intensity: common.ambient_light_intensity,
        shadow_map_resolution: common.shadow_map_resolution
    }
}

function GenerateLightingGrid(Viz) {
    const dummy_space_params = {
        shape: [10, 40, 10],
        size: 1.0,
        padding: 0.32,
        position: {x: 0, y:0, z:0},
        color: 0x808080,
        rotation:  GetDefault2dIterationPose()
    };
    const dummy_space = Viz.CreateIterationSpace(dummy_space_params);
    console.log(GetCommonConstants());
    const dummy_lighting = Viz.CreateIterationSpaceLighting({
        space: dummy_space,
        show_lights: false,
        top_light: {color: 0xffffff, panel_distance: 14, intensity : 0.5, distance : 35, decay : 2,  light_count: {x: 1, y: 1}, light_stride: {x: 12, y: 16}, offset: {x: 5, y: 0}},
        left_light: {color: 0xffffff, panel_distance: 13, intensity : 2.5, distance : 30, decay : 2,  light_count: {x: 3, y: 1}, light_stride: {x: 12, y: 10}, offset: {x: 25, y: 0}},
        right_light: {color: 0xffffff, panel_distance: 10, intensity : 0.3, distance : 40, decay : 2.2,  light_count: {x: 2, y: 2}, light_stride: {x: 18, y: 18}, offset: {x: 0, y: -5}},
   });

    dummy_lighting.root_object.position.x -= 15;
    dummy_space.remove();

    return dummy_lighting;
}

async function RunViz(Viz, SceneView) {
    const common = GetCommonConstants(); 

    const space1_params = {
        shape: [10, 10, 1],
        size: 1.0,
        padding: 0.15,
        position: {x: 0, y:0, z:0},
        color: common.fuse_color1,
        rotation:  {x :0.4, y: 2.85, z: 3.1}
    };

    const space2_params = {
        shape: [10, 10, 1],
        size: 1.0,
        padding: 0.15,
        position: {x: 0, y:0, z:0},
        color: common.fuse_color2,
        rotation:  {x :0.4, y: 2.85, z: 3.1}
    };

    let space_1_label = {
        axis0: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: 10, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "i0", 
            label_pos: 0.5, 
            label_size: 1
        },
        axis1: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: 10, 
            arrow_display_side: CUBE_SIDES.TOP,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "j0", 
            label_pos: 0.5, 
            label_size: 1
        },
    }

    let space_2_label = {
        axis0: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: 10, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "i1", 
            label_pos: 0.5, 
            label_size: 1
        },
        axis1: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: 10, 
            arrow_display_side: CUBE_SIDES.TOP,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "j1", 
            label_pos: 0.5, 
            label_size: 1
        },
    }


    let space_pair = Viz.CreateIterationSpacePair({
        space1_params: space1_params,
        space2_params: space2_params,
        space1_axis_label: space_1_label,
        space2_axis_label: space_2_label,
        spacing: 2,
        horizontal: true
    });

    Viz.camera.position.z = 45;
    Viz.camera.position.y = -1;
    Viz.camera.position.x = 6.5;
    Viz.camera.set_fov_zoom(0.6);

    let lighting = GenerateLightingGrid(Viz);

    await Viz.SaveImage("fuse1")
}

