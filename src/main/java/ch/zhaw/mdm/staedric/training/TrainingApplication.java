package ch.zhaw.mdm.staedric.training;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.Translator;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

@SpringBootApplication
public class TrainingApplication {

	// represents number of training samples processed before the model is updated
	private static final int BATCH_SIZE = 32;

	// the number of passes over the complete dataset
	private static final int EPOCHS = 2;

	public static void main(String[] args) throws IOException, TranslateException, MalformedModelException {

		/*
		 * ------------------- Aufruf des trainierten Models ----------------------------
		 * System.out.println(modelDir);
		 * System.out.println(Models.MODEL_NAME);
		 * System.out.println(predict());
		 */


		 //------------Training Model-----------------
		SpringApplication.run(TrainingApplication.class, args);
		// the location to save the model
		Path modelDir = Paths.get("models");
		// create ImageFolder dataset from directory
		ImageFolder dataset = initDataset("ut-zap50k-images-square");
		// Split the dataset set into training dataset and validate dataset
		RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);
		Loss loss = Loss.softmaxCrossEntropyLoss();

		// setting training parameters (ie hyperparameters)
		TrainingConfig config = setupTrainingConfig(loss);

		try (

				Model model = Models.getModel(); // empty model instance to hold patterns
				Trainer trainer = model.newTrainer(config)) {
			// metrics collect and report key performance indicators, like accuracy
			trainer.setMetrics(new Metrics());
			Shape inputShape = new Shape(1, 3, Models.IMAGE_HEIGHT, Models.IMAGE_HEIGHT);
			// initialize trainer with proper input shape
			trainer.initialize(inputShape);

			// find the patterns in data
			EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);

			// set model properties
			TrainingResult result = trainer.getTrainingResult();
			model.setProperty("Epoch", String.valueOf(EPOCHS));
			model.setProperty(
					"Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
			model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
			// save the model after done training for inference later
			// model saved as shoeclassifier-0000.params
			model.save(modelDir, Models.MODEL_NAME);
			// save labels into model directory
			Models.saveSynset(modelDir, dataset.getSynset());

		}
	}

	private static ImageFolder initDataset(String datasetRoot)
			throws IOException, TranslateException {
		ImageFolder dataset = ImageFolder.builder()
				// retrieve the data
				.setRepositoryPath(Paths.get(datasetRoot))
				.optMaxDepth(10)
				.addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
				.addTransform(new ToTensor())
				// random sampling; don't process the data in order
				.setSampling(BATCH_SIZE, true)
				.build();

		dataset.prepare();
		return dataset;
	}

	private static TrainingConfig setupTrainingConfig(Loss loss) {
		return new DefaultTrainingConfig(loss)
				.addEvaluator(new Accuracy())
				.addTrainingListeners(TrainingListener.Defaults.logging());
	}

	private static String predict() throws IOException, MalformedModelException, TranslateException {
		// the location where the model is saved
		Path modelDir = Paths.get("models");

		// the path of image to classify
		String imageFilePath;

		imageFilePath = "ut-zap50k-images-square/Sandals/Athletic/Keen/7596238.3.jpg";

		// Load the image file from the path
		Image img = ImageFactory.getInstance().fromFile(Paths.get(imageFilePath));

		try (Model model = Models.getModel()) { // empty model instance
			// load the model
			model.load(modelDir, Models.MODEL_NAME);

			// define a translator for pre and post processing
			// out of the box this translator converts images to ResNet friendly ResNet 18
			// shape
			Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
					.addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
					.addTransform(new ToTensor())
					.optApplySoftmax(true)
					.build();

			// run the inference using a Predictor
			try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
				// holds the probability score per label
				Classifications predictResult = predictor.predict(img);
				return predictResult.toString();
			}
		}

	}
}
