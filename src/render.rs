use bevy::{
    prelude::*,
    render::{
        pipeline::{Face, PipelineDescriptor, PrimitiveState, RenderPipeline},
        shader::{ShaderStage, ShaderStages},
    },
};
use ndarray::array;

use crate::filter::Particles;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(init_particles)
            .add_system(update_particles);
    }
}

struct ParticleMeshHandle(Handle<Mesh>);

fn init_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
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

    let mut particle_mesh = Mesh::new(bevy::render::pipeline::PrimitiveTopology::TriangleList);

    let vertexes = vec![[0.0, 0.0, 0.0]];
    particle_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, vertexes);

    let v_color = vec![[1.0, 0.1, 0.1]];
    particle_mesh.set_attribute("Vertex_Color", v_color);

    let mesh_handle = meshes.add(particle_mesh);

    commands.spawn_bundle(MeshBundle {
        mesh: mesh_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle,
        )]),
        ..Default::default()
    });

    commands.insert_resource(ParticleMeshHandle(mesh_handle));

    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
}

fn update_particles(
    mut particles: ResMut<Particles>,
    mesh_handle: Res<ParticleMeshHandle>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // lol
    let positions = vec![
        array![30.0, -30.0],
        array![30.0, 30.0],
        array![-30.0, -30.0],
    ];

    particles.predict(0.1, 1.0 / 60.0);

    // lol
    particles.update(1.0, &positions);

    if particles.neff() < (particles.n as f64 / 2.0) {
        particles.resample();
    }

    let estimated = particles.estimate(3);

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

    if let bevy::render::mesh::VertexAttributeValues::Float32x3(ref mut points) = colours {
        points.resize(particles.n, [1.0, 1.0, 1.0]);

        // TODO: hue

        // for (idx, pos) in particles.positions().columns().into_iter().enumerate() {
        //     points[idx] = [pos[0] as f32, pos[1] as f32, 0.0];
        // }
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

void main() {
    v_Color = Vertex_Color;
    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
    gl_PointSize = 5.0;
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
