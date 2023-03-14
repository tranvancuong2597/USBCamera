package com.jiangdg.usbcamera.view;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

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
    private static Classifier classifier;

    private Module mModule = null;

    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        imageView = findViewById(R.id.imageView);
        btnDetect = findViewById(R.id.detectButton);
        btnSelectFromGallery = findViewById(R.id.select_from_gallery);

        try {
            initClassifier();
        } catch (IOException e) {
            e.printStackTrace();
        }

        btnSelectFromGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openGallery();
            }
        });

        try {
            mModule = LiteModuleLoader.load(DetectionActivity.assetFilePath(getApplicationContext(), "resnet50_ver5.ptl"));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        btnDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d("Cuongcuong", "Vaoday");

                runInBackground(new Runnable() {
                    @Override
                    public void run() {
                        Integer cntAbnormal = 0;
                        try {
                            List<String> list_img = getImage(getBaseContext());
                            for (int i=0 ; i<list_img.size() ; i++) {
//                                InputStream inputStream = getAssets().open("abnormal/" + list_img.get(i));
                                Log.d("Cuongcuong", list_img.get(i));

                                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(list_img.get(i)));

                                cropBitmap = processBitmap(bitmap, 235);
                                cropBitmap = scaleCenterCrop(cropBitmap, 224, 224);
//                                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(cropBitmap, NO_MEAN_RGB, NO_STD_RGB);
//                                IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
//                                final Tensor outputTensor = outputTuple[0].toTensor();
//                                final float[] outputs = outputTensor.getDataAsFloatArray();
//                                Log.d("Cuongcuong: ", "output" + outputs);

//
//                                final float[][] results = classifier.recognizeImage_new(cropBitmap, "a");
//                                Log.d("Cuongcuong", "len results: " + results.length);
//                                Log.d("Cuongcuong", "results: " + results[0][0] + " " + results[0][1]);

//                                final ArrayList<Classifier.Recognition> detections = classifier.recognizeImage3(bitmap);

//                                if (results[0][0] > results[0][1]) {
//                                    cntAbnormal += 1;
//                                }

                            }
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
        String[] files = assetManager.list("abnormal");
        List<String> it= Arrays.asList(files);
        return it;
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }

        return bitmap;
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

    private void initClassifier() throws IOException {
        classifier = new Classifier(getAssets(), "resnet50-224.tflite","labels.txt",32, getBaseContext());
        classifier.init();
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



}