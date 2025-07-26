const express = require('express');
const fs = require('fs');

const app = express();
app.use(express.json());

const ort = require('onnxruntime-node');
const util = require('./util.js');
const models = {};

(async function () {
    await require('./models/download.js')(); // Ensure models are downloaded before loading
    
    const modelsDir = fs.readdirSync('models').filter(file => file.endsWith('.onnx'));

    for (const modelFile of modelsDir) {
        const fullPath = `models/${modelFile}`;

        models[modelFile] = await ort.InferenceSession.create(fullPath, {
            'interOpNumThreads': 12,
            'intraOpNumThreads': 4
        });

        console.log(`Model loaded: ${modelFile}`);
    }
})();

const dic = {
    'sheep': 'photosheap', // joke btw
    'pizza': 'photopizza'
}

app.post('/predict', async (req, res) => {
    let { variant, image } = req.body;

    if (!variant || !image) return res.status(400).send('Variant and image are required');

    variant = dic[variant] || variant;

    /** @type {ort.InferenceSession} */
    const model = models[`${variant}.onnx`];
    if (!model) return res.status(404).send('Model not found');

    try {
        const imageBuffer = Buffer.from(image, 'base64');
        const imgTensor = await util.preprocessImage(imageBuffer);
        const predict = await model.run({ images: imgTensor });

        const detections = util.parseNMSOutput(predict.output0, 0.5);
        if (!detections) return res.status(404).send('No detections found');

        const prediction = util.getGridIndexFromBox(detections);

        res.status(200).send({ prediction });
    } catch (error) {
        console.error('Error during inference:', error);
        res.status(500).send('Error during inference');
    }
});

const port = 8090;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});