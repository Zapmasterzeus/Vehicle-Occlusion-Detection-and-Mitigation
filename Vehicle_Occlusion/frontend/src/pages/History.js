import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

const History = () => {
  const [uploads, setUploads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useAuth();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        if (user?.isGuest) {
          setUploads([]);
          setLoading(false);
          return;
        }

        const response = await api.get('/uploads');
        setUploads(response.data.data || []);
      } catch (err) {
        console.error('Error fetching history:', err);
        setError(err.response?.data?.message || 'Failed to load upload history');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [user]);

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <CircularProgress />
        <Typography sx={{ mt: 2 }}>Loading history...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Upload History
      </Typography>

      {user?.isGuest && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Guest users don't have access to upload history. Please create an account or login.
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {!user?.isGuest && (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell>Upload Date</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Occlusion %</TableCell>
                <TableCell>Confidence</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {uploads.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    <Typography variant="body2" color="text.secondary">
                      No uploads found. Start by uploading your first image!
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                uploads.map((upload) => (
                  <TableRow key={upload.id}>
                    <TableCell>{upload.filename || 'Unknown'}</TableCell>
                    <TableCell>
                      {upload.uploadDate 
                        ? new Date(upload.uploadDate).toLocaleDateString()
                        : 'Unknown'
                      }
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={upload.status || 'Processed'}
                        color="success"
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{upload.occlusionPercentage || 0}%</TableCell>
                    <TableCell>{upload.confidenceScore || 0}%</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default History;