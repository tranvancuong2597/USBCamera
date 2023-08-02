package com.cuongtv.usbcamera.view;

import java.util.LinkedList;
import java.util.Queue;

public class ImageQueue {
    private Queue<ImageResult> imageResults;
    private int maxSize;

    public ImageQueue(int maxSize) {
        this.imageResults = new LinkedList<>();
        this.maxSize = maxSize;
    }

    public void addImageResult(Integer result) {
        // Xóa phần tử đầu tiên nếu Queue đã đạt đến kích thước tối đa
        if (imageResults.size() == maxSize) {
            imageResults.poll();
        }

        // Thêm kết quả hình ảnh mới vào Queue
        imageResults.offer(new ImageResult(result));
    }

    public Integer getFinalResult() {
        int normalCount = 0;
        int abnormalCount = 0;

        // Đếm số lượng hình ảnh bình thường và bất thường trong Queue
        for (ImageResult imageResult : imageResults) {
            if (imageResult.getResult().equals(1)) {
                normalCount++;
            } else {
                abnormalCount++;
            }
        }

        // Trả về kết quả dựa trên số lượng hình ảnh bình thường và bất thường
        if (normalCount < abnormalCount) {
            return 0;
        }
        return 1;
    }

}
