import React, { useEffect, useRef, useState } from 'react';
import { Box, Button, Stack, Typography, Alert } from '@mui/material';
import { uploadSegment } from '../services/stream';

function LiveStream() {
  const videoRef = useRef(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [running, setRunning] = useState(false);
  const [index, setIndex] = useState(1);
  const [sessionId] = useState(() => Math.random().toString(36).slice(2));
  const [lastResultUrl, setLastResultUrl] = useState(null);
  const [error, setError] = useState(null);
  const [hmin, setHmin] = useState(1);  // Start from 1st half-minute
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    return () => {
      try { mediaRecorder && mediaRecorder.state !== 'inactive' && mediaRecorder.stop(); } catch {}
      const v = videoRef.current;
      if (v && v.srcObject) {
        const tracks = v.srcObject.getTracks();
        tracks.forEach(t => t.stop());
        v.srcObject = null;
      }
    };
  }, [mediaRecorder]);

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const mr = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8' });
      mr.ondataavailable = async (e) => {
        if (!e.data || e.data.size === 0) return;
        const cur = index;
        setIndex(cur + 1);
        try {
          const { data } = await uploadSegment({ blob: e.data, index: cur, sessionId });
          const url = data?.data?.aodVideoUrl;
          if (url) {
            const abs = (process.env.REACT_APP_API_ORIGIN || 'http://localhost:5002') + url;
            setLastResultUrl(abs);
          }
        } catch (err) {
          // non-blocking error
          setError(err?.response?.data?.message || err.message);
        }
      };
      // timeslice of 1000ms creates 1s segments
      mr.start(1000);
      setMediaRecorder(mr);
      setRunning(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleUpload = async (blob, index) => {
    try {
      const { data } = await uploadSegment({ 
        blob, 
        index,
        sessionId 
      });
      
      if (data?.data?.hmin) {
        setHmin(data.data.hmin);
      }
      if (data?.data?.currentIndex) {
        setCurrentIndex(data.data.currentIndex);
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message);
    }
  };
  
  const stop = () => {
    try { mediaRecorder && mediaRecorder.stop(); } catch {}
    setRunning(false);
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Live Camera Stream (Half-Minute: {hmin}, Segment: {currentIndex % 60})</Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems="flex-start">
        <Box>
          <Typography variant="subtitle1">Camera Preview</Typography>
          <video ref={videoRef} style={{ width: 480, height: 270, background: '#000' }} muted playsInline />
          <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
            {!running ? (
              <Button variant="contained" onClick={start}>Start</Button>
            ) : (
              <Button variant="outlined" color="warning" onClick={stop}>Stop</Button>
            )}
          </Stack>
        </Box>
        <Box>
          <Typography variant="subtitle1">Last Processed Output</Typography>
          {lastResultUrl ? (
            <video src={lastResultUrl} style={{ width: 480, height: 270, background: '#000' }} controls />
          ) : (
            <Box sx={{ width: 480, height: 270, bgcolor: '#000' }} />
          )}
        </Box>
      </Stack>
    </Box>
  );
}

export default LiveStream;
