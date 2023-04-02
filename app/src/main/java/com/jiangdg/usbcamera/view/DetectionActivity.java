package com.jiangdg.usbcamera.view;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.jiangdg.usbcamera.R;
import com.jiangdg.usbcamera.tflite.Classifier;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

public class DetectionActivity extends AppCompatActivity implements Runnable{

    ImageView imageView;
    Button btnDetect, btnSelectFromGallery;

    private static final int PICK_IMAGE = 100;
    Uri imageUri;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;

    private Handler handler;
    private HandlerThread handlerThread;

    private long lastProcessingTimeMs;
    private long timestamp = 0;

    AssetManager assetManager = null;

    private static final int NUM_THREADS = 4;
    private static boolean isNNAPI = false;
    private static boolean isGPU = false;
    private Interpreter tfLite;
    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;
    private static final int INPUT_SIZE = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        imageView = findViewById(R.id.imageView);
        btnDetect = findViewById(R.id.detectButton);
        btnSelectFromGallery = findViewById(R.id.select_from_gallery);

        assetManager = getApplicationContext().getAssets();

        btnSelectFromGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openGallery();
            }
        });

        initTFLite("resnet50-224.tflite");

        btnDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                runInBackground(new Runnable() {
                    @Override
                    public void run() {
                        Integer cntAbnormal = 0;
                        Integer cntNormal = 0;
                        try {
                            List<String> list_img = getImage(getBaseContext());
                            for (int k=0 ; k<list_img.size() ; k++) {
//                                InputStream inputStream = getAssets().open("abnormal/" + list_img.get(i));
                                Log.d("Cuongcuong", list_img.get(k));
//                                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("abnormal/" + "aom (8).png"));
                                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("normal/" + list_img.get(k)));
                                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 235, 235, false);
                                cropBitmap = scaleCenterCrop(resizedBitmap, INPUT_SIZE, INPUT_SIZE);
                                float[][][] img = normalizeImage(cropBitmap);
                                float[][][][] input_net = new float[1][][][];
                                input_net[0] = img;
                                input_net = transposeBatch(input_net);

                                Map<Integer, Object> outputMap = new HashMap<>();
                                outputMap.put(0, new float[1][2]);
                                tfLite.runForMultipleInputsOutputs(new Object[]{input_net}, outputMap);
                                float[][] outputs = (float[][]) outputMap.get(0);
                                Log.d("Cuongcuong", outputs[0][0] + " " + outputs[0][1]);
                                if (outputs[0][0] > outputs[0][1]) {
                                    cntAbnormal += 1;
                                    Log.d("Cuongcuong", "abnormal");
                                } else {
                                    cntNormal += 1;
                                    Log.d("Cuongcuong", "normal");
                                }
                            }
                            Log.d("Cuongcuong: ", "cntNormal: " + cntNormal + " Ratio: " + cntNormal / list_img.size());
                            Log.d("Cuongcuong: ", "cntAbnormal: " + cntAbnormal + " Ratio: " + cntAbnormal / list_img.size());
                        } catch (IOException e) {
                            e.printStackTrace();
                        }


                    }
                });
            }
        });
    }

    @Override
    public void run() {

    }

    private List<String> getImage(Context context) throws IOException
    {
        AssetManager assetManager = context.getAssets();
        String[] files = assetManager.list("normal");
        List<String> it= Arrays.asList(files);
        return it;
    }

    private void openGallery() {
        Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE){
            imageUri = data.getData();
            //imageView.setImageURI(imageUri);
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                this.sourceBitmap = bitmap;
                this.cropBitmap = processBitmap(sourceBitmap, 248);
                this.imageView.setImageBitmap(sourceBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static Bitmap processBitmap(Bitmap source, int size){
        int image_height = source.getHeight();
        int image_width = source.getWidth();

        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);

        Matrix frameToCropTransformations = getTransformationMatrix(image_width,image_height,size,size,0,false);
        Matrix cropToFrameTransformations = new Matrix();
        frameToCropTransformations.invert(cropToFrameTransformations);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(source, frameToCropTransformations, null);

        return croppedBitmap;
    }

    public static Bitmap scaleCenterCrop(Bitmap source, int newHeight, int newWidth) {
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();

        float xScale = (float) newWidth / sourceWidth;
        float yScale = (float) newHeight / sourceHeight;
        float scale = Math.max(xScale, yScale);

        // Now get the size of the source bitmap when scaled
        float scaledWidth = scale * sourceWidth;
        float scaledHeight = scale * sourceHeight;

        float left = (newWidth - scaledWidth) / 2;
        float top = (newHeight - scaledHeight) / 2;

        RectF targetRect = new RectF(left, top, left + scaledWidth, top
                + scaledHeight);//from ww w  .j a va 2s. co m

        Bitmap dest = Bitmap.createBitmap(newWidth, newHeight,
                source.getConfig());
        Canvas canvas = new Canvas(dest);
        canvas.drawBitmap(source, null, targetRect, null);

        return dest;
    }

    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        handlerThread = new HandlerThread("inference2");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    protected void onPause() {
        super.onPause();
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.d("Cuongcuong", "onPause Exception");
        }
    }


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public void writeFileOnInternalStorage(Context mcoContext,String sFileName, String sBody){
        File file = new File(mcoContext.getFilesDir(),"mydir");
        String ff = mcoContext.getFilesDir().toString();
        if(!file.exists()){
            file.mkdir();
        }
        try{
            File gpxfile = new File(file, sFileName);
            BufferedWriter buf = new BufferedWriter(new FileWriter(gpxfile, true));
            buf.append(sBody);
            buf.newLine();
            buf.close();
        }catch (Exception e){
        }
    }

    // tfLite

    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void initTFLite(String modelFilename) {
        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                NnApiDelegate nnApiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    nnApiDelegate = new NnApiDelegate();
                    options.addDelegate(nnApiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(false);
                    options.setAllowFp16PrecisionForFp32(true);
                    options.setAllowBufferHandleOutput(true);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                GpuDelegate gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            }
            tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    public static float[][][] normalizeImage(Bitmap bitmap) {
        int h = bitmap.getHeight();
        int w = bitmap.getWidth();
        float[][][] floatValues = new float[h][w][3];

        float imageMean = 128;
        float imageStd = 128;

        int[] pixels = new int[h * w];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, w, h);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                final int val = pixels[i * w + j];
                float r = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                float g = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                float b = ((val & 0xFF) - imageMean) / imageStd;
                float[] arr = {r, g, b};
                floatValues[i][j] = arr;
            }
        }
        return floatValues;
    }

    public static float[][][][] transposeBatch(float[][][][] in) {
        // in b, h, w, c
        int batch = in.length;
        int h = in[0].length;
        int w = in[0][0].length;
        int channel = in[0][0][0].length;
        float[][][][] out = new float[batch][channel][w][h];
        for (int i = 0; i < batch; i++) {
            for (int j=0; j < channel ; j++) {
                for (int m = 0; m < h; m++) {
                    for (int n = 0; n < w; n++) {
                        out[i][j][n][m] = in[i][n][m][j] ;
                    }
                }
            }
        }
        return out;
    }

}