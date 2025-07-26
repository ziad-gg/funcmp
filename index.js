process.env.ORT_LOGGING_LEVEL = 'verbose';
process.env.ORT_NUM_THREADS = '4';
const ort = require('onnxruntime-node');
const sharp = require('sharp');

async function runOnImage(imagePath, modelPath) {
    const session = await ort.InferenceSession.create(modelPath, {
        'interOpNumThreads': 12,
        'intraOpNumThreads': 4
    });

    console.time('Inference Time');
    const inputTensor = await preprocessImage(imagePath);
    const feeds = { images: inputTensor };
    const results = await session.run(feeds);
    console.timeEnd('Inference Time');

    const detections = parseNMSOutput(results.output0, 0.5);

    console.log('Model output:', detections);

    const detection = detections;
    const index = getGridIndexFromBox(detection);

    console.log(`Grid index: ${index}`);
}

function getGridIndexFromBox(box, tileWidth = 100, tileHeight = 100, cols = 3, rows = 2) {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;

    const col = Math.min(Math.floor(centerX / tileWidth), cols - 1);
    const row = Math.min(Math.floor(centerY / tileHeight), rows - 1);

    return row * cols + col;
}

function parseNMSOutput(tensor, confidenceThreshold = 0.4) {
    const [batch, numBoxes, valuesPerBox] = tensor.dims;
    const data = tensor.data;

    let bestDetection = null;

    for (let i = 0; i < numBoxes; i++) {
        const offset = i * valuesPerBox;

        const x1 = data[offset];
        const y1 = data[offset + 1];
        const x2 = data[offset + 2];
        const y2 = data[offset + 3];
        const confidence = data[offset + 4];
        const classId = data[offset + 5];

        if (confidence > confidenceThreshold) {
            const det = {
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
                confidence,
                classId
            };

            if (!bestDetection || det.confidence > bestDetection.confidence) {
                bestDetection = det;
            }
        }
    }

    return bestDetection;
}

async function preprocessImage(imagePath) {
    const { data } = await sharp(imagePath)
        .resize(320, 320)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const floatArray = Float32Array.from(data).map(v => v / 255.0);

    const chw = new Float32Array(3 * 320 * 320);
    for (let i = 0; i < 320 * 320; i++) {
        chw[i] = floatArray[i * 3];                     // R
        chw[i + 320 * 320] = floatArray[i * 3 + 1];     // G
        chw[i + 2 * 320 * 320] = floatArray[i * 3 + 2]; // B
    }

    return new ort.Tensor('float32', chw, [1, 3, 320, 320]);
}

// photosheap image and model paths
// const imagePath = 'data/variants/photosheap/image_2.jpg';
// const modelPath = 'models/photosheap.onnx';
    
// photopizza image and model paths
const imagePath = 'data/variants/photopizza/image_1.jpg';
const modelPath = 'models/photopizza.onnx';

runOnImage(imagePath, modelPath);