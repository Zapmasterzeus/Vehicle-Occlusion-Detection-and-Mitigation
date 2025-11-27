const express = require('express');
const multer = require('multer');
const fs = require('fs-extra');
const path = require('path');
const { spawn } = require('child_process');

const router = express.Router();
const sessionTimers = new Map();
// Project root is 3 levels up from backend/src
const projectRoot = path.resolve(__dirname, '../../..');
const segmentsRoot = path.join(projectRoot, 'segments');
fs.mkdirSync(segmentsRoot, { recursive: true });

// Optional ffmpeg conversion support
let ffmpeg;
let ffmpegPath;
try {
  ffmpeg = require('fluent-ffmpeg');
  ffmpegPath = require('ffmpeg-static');
  if (ffmpegPath) {
    ffmpeg.setFfmpegPath(ffmpegPath);
  }
} catch (e) {
  // Conversion may be unavailable; we will proceed without it
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const sessionId = (req.body.sessionId || 'default').replace(/[^a-zA-Z0-9_-]/g, '_');
    const dir = path.join(segmentsRoot, sessionId);
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const idx = req.body.index ? parseInt(req.body.index, 10) : Date.now();
    // Save with original extension; we may convert after
    const ext = path.extname(file.originalname) || (file.mimetype === 'video/mp4' ? '.mp4' : '.webm');
    cb(null, `sec_${idx}${ext}`);
  }
});

const upload = multer({ storage });

async function cleanupOldSegments(sessionId, currentIndex) {
    const sessionDir = path.join(segmentsRoot, sessionId);
    if (!fs.existsSync(sessionDir)) return;

    // Calculate threshold (30 seconds ago)
    const threshold = 30; // seconds
    const thresholdIndex = currentIndex - threshold;

    try {
        const files = await fs.readdir(sessionDir);
        for (const file of files) {
            const match = file.match(/sec_(\d+)/);
            if (match) {
                const fileIndex = parseInt(match[1], 10);
                if (fileIndex <= thresholdIndex) {
                    await fs.remove(path.join(sessionDir, file));
                    // Also clean up output if it exists
                    const outputPath = path.join(projectRoot, 'outputs_vid', 'aod', 'vid', `sec_${fileIndex}.mp4`);
                    if (await fs.pathExists(outputPath)) {
                        await fs.remove(outputPath);
                    }
                }
            }
        }
    } catch (err) {
        console.error('Cleanup error:', err);
    }
}

async function cleanupEmptyDirs(dir) {
    try {
        const files = await fs.readdir(dir);
        if (files.length === 0) {
            await fs.remove(dir);
            return true;
        }
        
        let allRemoved = true;
        for (const file of files) {
            const fullPath = path.join(dir, file);
            const stat = await fs.stat(fullPath);
            if (stat.isDirectory()) {
                const removed = await cleanupEmptyDirs(fullPath);
                if (!removed) allRemoved = false;
            } else {
                allRemoved = false;
            }
        }
        
        if (allRemoved) {
            await fs.remove(dir);
            return true;
        }
        return false;
    } catch (err) {
        console.error('Cleanup dirs error:', err);
        return false;
    }
}

setInterval(async () => {
    try {
        await cleanupEmptyDirs(segmentsRoot);
        await cleanupEmptyDirs(path.join(projectRoot, 'outputs_vid', 'aod', 'vid'));
    } catch (err) {
        console.error('Periodic cleanup error:', err);
    }
}, 5 * 60 * 1000);

function findPythonExe() {
  const venvPy = path.join(projectRoot, '.venv', 'Scripts', process.platform === 'win32' ? 'python.exe' : 'python');
  if (fs.existsSync(venvPy)) return venvPy;
  return process.env.PYTHON_EXEC || 'python';
}

function runFinal(segmentAbs, sessionId) {
  return new Promise((resolve, reject) => {
    const py = findPythonExe();
    const finalRun = path.join(projectRoot, 'final_run.py');
    const proc = spawn(py, [finalRun, '--segment', segmentAbs, '--session', sessionId], {
      cwd: projectRoot,
      env: { ...process.env },
      shell: false
    });

    let parsed = null;
    proc.stdout.on('data', (buf) => {
      const s = buf.toString();
      const lines = s.split(/\r?\n/);
      for (const line of lines) {
        const i = line.indexOf('AOD_RESULT:');
        if (i >= 0) {
          const jsonPart = line.slice(i + 'AOD_RESULT:'.length).trim();
          try {
            parsed = JSON.parse(jsonPart);
          } catch {}
        }
      }
    });
    proc.stderr.on('data', () => {});
    proc.on('close', (code) => {
      if (parsed) return resolve(parsed);
      if (code === 0) return resolve({ status: 'ok', aodVideoUrl: null });
      reject(new Error(`final_run exited with code ${code}`));
    });
  });
}

async function maybeConvertToMp4(srcPath) {
  const ext = path.extname(srcPath).toLowerCase();
  if (ext === '.mp4') return srcPath;
  if (!ffmpeg || !ffmpegPath) return srcPath; // no converter available
  return await new Promise((resolve, reject) => {
    const dstPath = srcPath.replace(/\.[^.]+$/, '.mp4');
    ffmpeg(srcPath)
      .outputOptions(['-movflags +faststart', '-pix_fmt yuv420p'])
      .toFormat('mp4')
      .on('end', () => resolve(dstPath))
      .on('error', (err) => resolve(srcPath))
      .save(dstPath);
  });
}

// POST /api/stream/segment
router.post('/segment', upload.single('segment'), async (req, res) => {
    try {
        const sessionId = (req.body.sessionId || 'default').replace(/[^a-zA-Z0-9_-]/g, '_');
        const index = parseInt(req.body.index, 10) || 0;
        
        // Initialize or update session timer
        if (!sessionTimers.has(sessionId)) {
            sessionTimers.set(sessionId, {
                hmin: 0,
                lastCleanupIndex: 0
            });
        }
        
        const session = sessionTimers.get(sessionId);
        
        // Increment half-minute counter every 30 seconds
        if (index % 30 === 0 && index > session.lastCleanupIndex) {
            session.hmin++;
            session.lastCleanupIndex = index;
        }
        
        // Cleanup old segments
        await cleanupOldSegments(sessionId, index);

        // Rest of the existing code...
        const result = await runFinal(segmentPath, sessionId, session.hmin);
        return res.json({ 
            success: true, 
            data: {
                ...result.data,
                hmin: session.hmin,
                currentIndex: index
            }
        });
        // ...
    } catch (err) {
        console.error('Segment processing error:', err);
        return res.status(500).json({ 
            success: false, 
            message: err.message,
            hmin: session?.hmin || 0
        });
    }
});

module.exports = router;
