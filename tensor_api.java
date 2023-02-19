import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class MyApi {

  private final SavedModelBundle model;
  private final Session session;

  public MyApi(String ckptPath) {
    
    model = SavedModelBundle.load(ckptPath, "serve");

    session = model.session();
  }

  public float predict(float[] input) {
    
    Tensor<Float> inputTensor = Tensor.create(input, Float.class);

    Tensor<Float> outputTensor = session.runner()
        .feed("input", inputTensor)
        .fetch("output")
        .run()
        .get(0)
        .expect(Float.class);

    float output = outputTensor.floatValue();

    inputTensor.close();
    outputTensor.close();

    return output;
  }
}