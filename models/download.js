console.clear();

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cliProgress = require('cli-progress');

const baseUrl = 'https://github.com/ziad-gg/funcmp/releases/download/models/';
const requireFile = 'require.txt';

async function downloadFile(url, dest, label = '') {
    const writer = fs.createWriteStream(dest);

    const { data, headers } = await axios({
        url,
        method: 'GET',
        responseType: 'stream',
        maxRedirects: 5,
        validateStatus: status => status < 400
    });

    const totalSize = parseInt(headers['content-length'], 10);
    let downloaded = 0;

    const bar = new cliProgress.SingleBar({
        format: `${label} [{bar}] {percentage}%`,
        clearOnComplete: true,
        barsize: 30
    });

    bar.start(100, 0);

    data.on('data', chunk => {
        downloaded += chunk.length;
        const percent = Math.min((downloaded / totalSize) * 100, 100);
        bar.update(percent);
    });

    data.pipe(writer);

    return new Promise((resolve, reject) => {
        writer.on('finish', () => {
            bar.update(100);
            bar.stop();
            resolve();
        });
        writer.on('error', err => {
            bar.stop();
            reject(err);
        });
    });
}

async function main() {
    const requireUrl = baseUrl + requireFile;
    const requirePath = path.join(__dirname, requireFile);

    if (!fs.existsSync(requirePath)) {
        // console.log(`Downloading ${requireFile}...`);
        await downloadFile(requireUrl, requirePath, 'require.txt');
    }

    const modelNames = fs.readFileSync(requirePath, 'utf-8')
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean);

    for (const modelName of modelNames) {
        const modelPath = path.join(__dirname, modelName);
        if (fs.existsSync(modelPath)) continue;
        
        const modelUrl = baseUrl + modelName;
        try {
            await downloadFile(modelUrl, modelPath, modelName);
        } catch (err) {
            console.error(`Failed to download ${modelName}: ${err.message}`);
        }
    }
}

module.exports = main;
// main().catch(console.error);
