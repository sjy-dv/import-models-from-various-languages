use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    std::fs::File::open("model.ckpt")?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;

    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    let input_data = Tensor::<f32>::new(&[1, 2]).with_values(&[1.0, 2.0])?;

    let output_tensor = session.run(&[("input", &input_data)], &["output"])?[0].clone();

    let output_values = output_tensor.to::<f32>()?;
    println!("Output values: {:?}", output_values);

    Ok(())
}
