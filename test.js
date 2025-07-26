const { default: axios } = require('axios');
const fs = require('fs');

async function main(variant) {
    const variantDir = `data/variants/${variant}`;
    const files = fs.readdirSync(variantDir);
    const randomImage = files[Math.floor(Math.random() * files.length)];

    console.log(`Selected image: ${randomImage}`);

    const imagePath = `${variantDir}/${randomImage}`;
    const imageBuffer = fs.readFileSync(imagePath, 'base64');

    const response = await axios.post('http://localhost:8090/predict', {
        variant,
        image: imageBuffer
    });

    console.log(`Response from server: ${JSON.stringify(response.data)}`);
}

main('photosheap').catch(console.error);