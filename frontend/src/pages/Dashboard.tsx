import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
} from '@mui/material';
import axios from 'axios';

interface TenantUsage {
  users: { current: number; limit: number };
  models: { current: number; limit: number };
  storage_gb: { current: number; limit: number };
}

export default function Dashboard() {
  const [usage, setUsage] = useState<TenantUsage | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUsage();
  }, []);

  const fetchUsage = async () => {
    try {
      const response = await axios.get('/api/v1/tenants/usage');
      setUsage(response.data);
    } catch (error) {
      console.error('Error fetching usage:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Welcome to your Surrogate Model Platform. Here's an overview of your current usage.
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Users
              </Typography>
              <Typography variant="h3" color="primary">
                {usage?.users.current || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                of {usage?.users.limit} users
              </Typography>
              <LinearProgress
                variant="determinate"
                value={usage ? (usage.users.current / usage.users.limit) * 100 : 0}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Models
              </Typography>
              <Typography variant="h3" color="primary">
                {usage?.models.current || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                of {usage?.models.limit} models
              </Typography>
              <LinearProgress
                variant="determinate"
                value={usage ? (usage.models.current / usage.models.limit) * 100 : 0}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Storage
              </Typography>
              <Typography variant="h3" color="primary">
                {usage?.storage_gb.current || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                of {usage?.storage_gb.limit} GB
              </Typography>
              <LinearProgress
                variant="determinate"
                value={usage ? (usage.storage_gb.current / usage.storage_gb.limit) * 100 : 0}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Start Guide
              </Typography>
              <Typography variant="body1" paragraph>
                Get started with your surrogate modeling workflow:
              </Typography>
              <Box component="ol" sx={{ pl: 2 }}>
                <Typography component="li" sx={{ mb: 1 }}>
                  <strong>Upload Data:</strong> Go to Datasets and upload your simulation results (CSV or Excel format)
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  <strong>Train Model:</strong> Navigate to Models and create a new surrogate model from your dataset
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  <strong>Make Predictions:</strong> Use the Predictions page to get fast results from your trained model
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  <strong>Integrate:</strong> Use our API for programmatic access to predictions
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}