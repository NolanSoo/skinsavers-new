const fs = require('fs');
const path = require('path');

const IMAGE_DIR = path.join(__dirname, 'images');
const stages = ['benign', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5'];

function mapImagesToLabels() {
  const data = [];
  stages.forEach((stage, stageIdx) => {
    const stageDir = path.join(IMAGE_DIR, stage);
    if (fs.existsSync(stageDir)) {
      fs.readdirSync(stageDir).forEach(file => {
        if (file.endsWith('.jpg') || file.endsWith('.png')) {
          data.push({
            imagePath: path.join(stageDir, file),
            stage, // 'benign', 'stage1', ...
            stageLabel: stageIdx // 0 for benign, 1 for stage1, // 2 for stage2, etc.  
          });
        }
      });
    }
  });
  return data;
}

const mapped = mapImagesToLabels();     
console.log(mapped);