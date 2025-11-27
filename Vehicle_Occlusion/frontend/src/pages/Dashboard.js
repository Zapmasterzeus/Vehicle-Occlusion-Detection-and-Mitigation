import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Paper,
  Chip,
  Alert,
} from '@mui/material';
import {
  CloudUpload,
  Assessment,
  History,
  TrendingUp,
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalUploads: 0,
    totalDetections: 0,
    recentUploads: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        if (user?.isGuest) {
          // For guest users, show demo data
          setStats({
            totalUploads: 0,
            totalDetections: 0,
            recentUploads: [],
          });
          setLoading(false);
          return;
        }

        // For authenticated users, fetch real data
        const [uploadsRes, detectionsRes] = await Promise.all([
          api.get('/uploads'),
          api.get('/detections'),
        ]);

        setStats({
          totalUploads: uploadsRes.data.data.length,
          totalDetections: detectionsRes.data.data.length,
          recentUploads: uploadsRes.data.data.slice(0, 5),
        });
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [user]);

  const handleUploadClick = () => {
    navigate('/upload');
  };

  const handleHistoryClick = () => {
    navigate('/history');
  };

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Typography>Loading dashboard...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Welcome back, {user?.firstName || 'Guest'}!
          {user?.isGuest && (
            <Chip 
              label="Guest Mode" 
              color="secondary" 
              size="small" 
              sx={{ ml: 2 }} 
            />
          )}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Vehicle Occlusion Detection Dashboard
        </Typography>
      </Box>

      {user?.isGuest && (
        <Alert severity="info" sx={{ mb: 3 }}>
          You're using guest mode. Uploads will show demo results for demonstration purposes. Consider creating an account for full functionality.
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CloudUpload color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Total Uploads</Typography>
              </Box>
              <Typography variant="h3" component="div">
                {stats.totalUploads}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Assessment color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Detections</Typography>
              </Box>
              <Typography variant="h3" component="div">
                {stats.totalDetections}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUp color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Success Rate</Typography>
              </Box>
              <Typography variant="h3" component="div">
                {stats.totalUploads > 0 
                  ? Math.round((stats.totalDetections / stats.totalUploads) * 100)
                  : 0}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <History color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Recent Activity</Typography>
              </Box>
              <Typography variant="h3" component="div">
                {stats.recentUploads.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={<CloudUpload />}
                  onClick={handleUploadClick}
                  sx={{ py: 2 }}
                >
                  Upload New Image
                </Button>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<History />}
                  onClick={handleHistoryClick}
                  sx={{ py: 2 }}
                >
                  View History
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Recent Uploads */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Uploads
            </Typography>
            {stats.recentUploads.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                No uploads yet. Start by uploading your first image!
              </Typography>
            ) : (
              <Box sx={{ mt: 2 }}>
                {stats.recentUploads.map((upload, index) => (
                  <Box key={upload.id || index} sx={{ mb: 2 }}>
                    <Typography variant="body2" fontWeight="medium">
                      {upload.filename || `Upload ${index + 1}`}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {upload.uploadDate ? new Date(upload.uploadDate).toLocaleDateString() : 'Recently'}
                    </Typography>
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;