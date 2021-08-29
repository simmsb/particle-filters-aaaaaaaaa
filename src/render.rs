use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::{
        pipeline::{Face, PipelineDescriptor, PrimitiveState, RenderPipeline},
        render_graph::{base, AssetRenderResourcesNode, RenderGraph},
        renderer::RenderResources,
        shader::{ShaderStage, ShaderStages},
    },
};
use itertools::Itertools;
use ndarray::array;

use crate::filter::Particles;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(init_particles)
            .add_asset::<PointSize>()
            .add_system(update_particles);
    }
}

struct ParticleMeshHandle(Handle<Mesh>);
struct LandmarkMeshHandle(Handle<Mesh>);

#[derive(RenderResources, Default, TypeUuid)]
#[uuid = "ae94b745-c056-44f7-89e7-f26cad57df40"]
struct PointSize {
    value: f32,
}

fn init_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut point_sizes: ResMut<Assets<PointSize>>,
    mut render_graph: ResMut<RenderGraph>,
) {
    let pipeline_handle = pipelines.add(PipelineDescriptor {
        primitive: PrimitiveState {
            topology: bevy::render::pipeline::PrimitiveTopology::PointList,
            strip_index_format: None,
            front_face: bevy::render::pipeline::FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            polygon_mode: bevy::render::pipeline::PolygonMode::Point,
            clamp_depth: false,
            conservative: false,
        },
        ..PipelineDescriptor::default_config(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(ShaderStage::Vertex, VERTEX_SHADER)),
            fragment: Some(shaders.add(Shader::from_glsl(ShaderStage::Fragment, FRAGMENT_SHADER))),
        })
    });

    render_graph.add_system_node(
        "point_size",
        AssetRenderResourcesNode::<PointSize>::new(true),
    );

    render_graph
        .add_node_edge("point_size", base::node::MAIN_PASS)
        .unwrap();

    do_particles(
        &mut meshes,
        &mut commands,
        &mut point_sizes,
        pipeline_handle.clone(),
    );

    do_landmarks(
        &mut meshes,
        &mut commands,
        &mut point_sizes,
        pipeline_handle,
    );

    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
}

fn do_landmarks(
    meshes: &mut ResMut<Assets<Mesh>>,
    commands: &mut Commands,
    point_sizes: &mut ResMut<Assets<PointSize>>,
    pipeline_handle: Handle<PipelineDescriptor>,
) {
    let mut landmark_mesh = Mesh::new(bevy::render::pipeline::PrimitiveTopology::TriangleList);

    let vertexes = vec![[50.0, -30.0, 0.0], [30.0, 60.0, 0.0], [-70.0, -30.0, 0.0], [0.0, 0.0, 0.0]];
    let v_color = vec![[1.0, 0.1, 0.1]; vertexes.len()];

    landmark_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertexes);
    landmark_mesh.set_attribute("Vertex_Color", v_color);

    let landmark_mesh_handle = meshes.add(landmark_mesh);

    let point_size = point_sizes.add(PointSize { value: 7.0 });

    commands
        .spawn_bundle(MeshBundle {
            mesh: landmark_mesh_handle.clone(),
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                pipeline_handle,
            )]),
            ..Default::default()
        })
        .insert(point_size);

    commands.insert_resource(LandmarkMeshHandle(landmark_mesh_handle));
}

fn do_particles(
    meshes: &mut ResMut<Assets<Mesh>>,
    commands: &mut Commands,
    point_sizes: &mut ResMut<Assets<PointSize>>,
    pipeline_handle: Handle<PipelineDescriptor>,
) {
    let mut particle_mesh = Mesh::new(bevy::render::pipeline::PrimitiveTopology::TriangleList);

    let vertexes = vec![[0.0, 0.0, 0.0]];
    particle_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertexes);

    let v_color = vec![[1.0, 0.1, 0.1]];
    particle_mesh.set_attribute("Vertex_Color", v_color);

    let particle_mesh_handle = meshes.add(particle_mesh);

    let point_size = point_sizes.add(PointSize { value: 4.0 });

    commands
        .spawn_bundle(MeshBundle {
            mesh: particle_mesh_handle.clone(),
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                pipeline_handle,
            )]),
            ..Default::default()
        })
        .insert(point_size);

    commands.insert_resource(ParticleMeshHandle(particle_mesh_handle));
}

fn update_particles(
    mut particles: ResMut<Particles>,
    mesh_handle: Res<ParticleMeshHandle>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // lol
    let positions = vec![
        array![50.0, -30.0],
        array![30.0, 60.0],
        array![-70.0, -30.0],
        array![0.0, 0.0],
    ];

    particles.predict(1.0, 1.0 / 10.0);

    // lol
    particles.update(20.0, &positions);

    if particles.neff() < (particles.n as f32 / 6.0) {
        println!("resampling");
        dbg!(particles.neff());
        particles.resample();
        let _estimated = particles.estimate(positions.len());
    }

    // let estimated = particles.estimate(3);

    let particle_mesh = meshes.get_mut(&mesh_handle.0).unwrap();
    let vertexes = particle_mesh
        .attribute_mut(Mesh::ATTRIBUTE_POSITION)
        .unwrap();

    // let (x_bound, y_bound) = particles.upper_bounds();
    // let (x_offs, y_offs) = (x_bound as f32 / 2.0, y_bound as f32 / 2.0);
    let (x_offs, y_offs) = (0.0, 0.0);

    if let bevy::render::mesh::VertexAttributeValues::Float32x3(ref mut points) = vertexes {
        points.resize(particles.n, [0.0, 0.0, 0.0]);

        for (idx, pos) in particles.positions().rows().into_iter().enumerate() {
            points[idx] = [pos[0] as f32 + x_offs, pos[1] as f32 + y_offs, 0.0];
        }
    } else {
        panic!("huh")
    }

    let colours = particle_mesh.attribute_mut("Vertex_Color").unwrap();

    let max_colour = *particles.latest_groups().iter().max().unwrap() as f64;
    let (&min_weight, &max_weight) = particles.weights().iter().minmax().into_option().unwrap();
    let weight_range = max_weight - min_weight;

    if let bevy::render::mesh::VertexAttributeValues::Float32x3(ref mut colours) = colours {
        use palette::FromColor;

        colours.resize(particles.n, [1.0, 1.0, 1.0]);

        for (idx, (weight, grp)) in
            itertools::izip!(particles.weights(), particles.latest_groups()).enumerate()
        {
            let val = ((*weight - min_weight) / weight_range).abs() * 0.9 + 0.1;
            let col = palette::rgb::Rgb::from_color(palette::Hsv::new(
                palette::RgbHue::from_degrees(360.0 * *grp as f64 / (max_colour + 1.0)),
                1.0,
                val as f64,
            ));
            colours[idx] = [col.red as f32, col.green as f32, col.blue as f32];
        }
    } else {
        panic!("huh")
    }
}

const VERTEX_SHADER: &str = r"
#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Color;

layout(location = 1) out vec3 v_Color;

layout(set = 0, binding = 0) uniform CameraViewProj {
    mat4 ViewProj;
};

layout(set = 1, binding = 0) uniform Transform {
    mat4 Model;
};

layout(set = 2, binding = 0) uniform PointSize_value {
    float point_size;
};

void main() {
    v_Color = Vertex_Color;
    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
    gl_PointSize = point_size;
}
";

const FRAGMENT_SHADER: &str = r"
#version 450

layout(location = 1) in vec3 v_Color;
layout(location = 0) out vec4 o_Target;

void main() {
    o_Target = vec4(v_Color, 1.0);
}
";
