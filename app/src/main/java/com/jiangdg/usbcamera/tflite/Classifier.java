package com.jiangdg.usbcamera.tflite;


import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import com.jiangdg.usbcamera.ml.Levit224;
import com.jiangdg.usbcamera.ml.Resnet50224;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Classifier {
    private AssetManager assetManager;
    private String modelPath;
    private String labelPath;
    private List<String> labelList;
    private int inputSize = 32;

    private static final int INPUT_SIZE = 224;
    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    private Interpreter tfLite;
    private Context context;

    public Classifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize, Context context) {
        this.assetManager = assetManager;
        this.modelPath = modelPath;
        this.labelPath = labelPath;
        this.inputSize = inputSize;
        this.context = context;
    }

    public class Recognition{
        private String id = "";
        private String title = "";
        private float confidence = 0f;

        public Recognition(String id, String title, float confidence) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return "Pred:{" +
                    "title=" + title +
                    ", confidence=" + confidence +
                    '}';
        }
    }

    public void init() throws IOException {
        Interpreter.Options options= new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);

        tfLite = new Interpreter(loadModelFile(assetManager, modelPath),options);
        labelList = loadLabelList(assetManager, labelPath);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }


    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) {
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        Log.d("Cuongcuong", bitmap.getWidth() + " " + bitmap.getHeight());
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap, String filename) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

        ArrayList<Recognition> detections;

        detections = getDetection(byteBuffer, bitmap);
        return detections;
    }

    private ArrayList<Recognition> getDetection(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][2]);
        Object[] inputArray = {byteBuffer};
        //Log.d("Cuongcuong", "inputArray" + String.valueOf(inputArray));
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        float[][] output = (float [][]) outputMap.get(0);
        Log.d("Cuongcuong", output.length + "");
        Log.d("Cuongcuong", output[0][0] + " " + output[0][1]);
        return detections;
    }

    public float[][] recognizeImage_new(Bitmap bitmap, String filename) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

        float[][] detections;

        detections = getDetection_new(byteBuffer, bitmap);
        return detections;
    }

    private float[][] getDetection_new(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][2]);
        Object[] inputArray = {byteBuffer};
        //Log.d("Cuongcuong", "inputArray" + String.valueOf(inputArray));
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        float[][] output = (float [][]) outputMap.get(0);
        Log.d("Cuongcuong", output.length + "");
        Log.d("Cuongcuong", output[0][0] + " " + output[0][1]);
        return output;
    }

    public ArrayList<Recognition> recognizeImage3(Bitmap bitmap) {
        float[] mean = {(float) 0.5, (float) 0.5, (float) 0.5};
        float[] std = {(float) 0.5, (float) 0.5, (float) 0.5};

        ImageProcessor processor = new ImageProcessor.Builder()
                .add(new ResizeOp(235, 235, ResizeOp.ResizeMethod.BILINEAR))
                .add(new ResizeWithCropOrPadOp(224, 224))
                .add(new NormalizeOp(Float.valueOf(127.5F), Float.valueOf(127.5F)))
                //.add(new NormalizeOp(mean, std))
                .build();

        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        tensorImage.load(bitmap);
        tensorImage = processor.process(tensorImage);

        Bitmap tmpBitmap = tensorImage.getBitmap();
        Log.d("Cuongcuong", tensorImage.getHeight() + " " + tensorImage.getWidth() + "");


        ArrayList<Recognition> detections;
        detections = getDetection3(tensorImage.getBuffer());
        return detections;
    }

    private ArrayList<Recognition> getDetection3(ByteBuffer byteBuffer) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        try {
            Resnet50224 model = Resnet50224.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 3, 224, 224}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Resnet50224.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] data=outputFeature0.getFloatArray();

            Log.d("Cuongcuong", "data: " + data[0] + " " + data[1]);
            Log.d("Cuongcuong", "Cuongcuong");

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

        return detections;
    }

    public void writeData(String mtext, float[] output_data) {
        try {
            String filename = Environment.getExternalStorageDirectory()
                    .getAbsolutePath() + "/output4.txt";
            Log.d("Cuongcuong", filename);
            FileWriter fw = new FileWriter(filename, true);
            fw.write(mtext + " ");
            for (int i=0 ; i<output_data.length ; i++) {
                fw.write(output_data[i] + " ");
            }
            fw.write("\n");
            fw.close();
        } catch (IOException ioe) {
        }

    }



}

