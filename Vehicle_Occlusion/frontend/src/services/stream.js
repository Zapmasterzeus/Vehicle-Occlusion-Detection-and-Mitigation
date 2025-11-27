import api from './api';

export async function uploadSegment({ blob, index, sessionId }) {
  const form = new FormData();
  const filename = `sec_${index}.webm`;
  form.append('segment', blob, filename);
  form.append('index', String(index));
  form.append('sessionId', sessionId);

  const res = await api.post('/stream/segment', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
}
