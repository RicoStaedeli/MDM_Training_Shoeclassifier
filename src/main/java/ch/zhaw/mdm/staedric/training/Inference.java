package ch.zhaw.mdm.staedric.training;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

/** Uses the model to generate a prediction called an inference */
public class Inference {

    public Inference(){

    }

    public Classifications predict(byte[] image) throws IOException, ModelException, TranslateException {
        InputStream is = new ByteArrayInputStream(image);
        BufferedImage bi = ImageIO.read(is);
        Image img = ImageFactory.getInstance().fromImage(bi);

        // the location where the model is saved
        Path modelDir = Paths.get("models");

        try (Model model = Models.getModel()) { // empty model instance
            // load the model
            model.load(modelDir, Models.MODEL_NAME);

            // define a translator for pre and post processing
            // out of the box this translator converts images to ResNet friendly ResNet 18 shape
            Translator<Image, Classifications> translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                            .addTransform(new ToTensor())
                            .optApplySoftmax(true)
                            .build();

            // run the inference using a Predictor
            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                // holds the probability score per label
                Classifications predictResult = predictor.predict(img);
                System.out.println(predictResult);
                return predictResult;
            }
        }
    }
}