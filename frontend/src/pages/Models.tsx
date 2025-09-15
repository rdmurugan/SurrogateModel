import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  LinearProgress,
  Alert,
} from '@mui/material';
import { Add, Psychology } from '@mui/icons-material';
import axios from 'axios';

interface Model {
  id: number;
  name: string;
  algorithm: string;
  training_status: string;
  is_deployed: boolean;
  validation_metrics: any;
  created_at: string;
}

interface Dataset {
  id: number;
  name: string;
}

export default function Models() {
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    dataset_id: '',
    algorithm: 'gaussian_process',
  });
  const [createLoading, setCreateLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchModels();
    fetchDatasets();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/v1/models/');
      setModels(response.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchDatasets = async () => {
    try {
      const response = await axios.get('/api/v1/datasets/');
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const handleCreate = async () => {
    if (!createForm.name || !createForm.dataset_id) {
      setError('Please fill in all required fields');
      return;
    }

    setCreateLoading(true);
    setError('');

    try {
      await axios.post('/api/v1/models/', {
        name: createForm.name,
        description: createForm.description,
        dataset_id: parseInt(createForm.dataset_id),
        algorithm: createForm.algorithm,
      });

      setCreateOpen(false);
      setCreateForm({
        name: '',
        description: '',
        dataset_id: '',
        algorithm: 'gaussian_process',
      });
      fetchModels();
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Model creation failed');
    } finally {
      setCreateLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'training': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Surrogate Models</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateOpen(true)}
          disabled={datasets.length === 0}
        >
          Create Model
        </Button>
      </Box>

      {datasets.length === 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          You need to upload a dataset first before creating models.
        </Alert>
      )}

      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} md={6} key={model.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {model.name}
                </Typography>
                <Box display="flex" gap={1} mb={2}>
                  <Chip
                    label={model.training_status}
                    color={getStatusColor(model.training_status)}
                    size="small"
                  />
                  <Chip label={model.algorithm} size="small" />
                  {model.is_deployed && (
                    <Chip label="Deployed" color="success" size="small" />
                  )}
                </Box>

                {model.validation_metrics && (
                  <Box>
                    <Typography variant="body2" gutterBottom>
                      <strong>Performance Metrics:</strong>
                    </Typography>
                    <Typography variant="body2">
                      R² Score: {model.validation_metrics.r2_score}
                    </Typography>
                    <Typography variant="body2">
                      RMSE: {model.validation_metrics.rmse}
                    </Typography>
                    <Typography variant="body2">
                      MAE: {model.validation_metrics.mae}
                    </Typography>
                  </Box>
                )}

                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Created: {new Date(model.created_at).toLocaleDateString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {models.length === 0 && datasets.length > 0 && (
        <Card sx={{ mt: 3 }}>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Psychology sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No models created yet
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Create your first surrogate model from your uploaded datasets
            </Typography>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setCreateOpen(true)}
              sx={{ mt: 2 }}
            >
              Create Your First Model
            </Button>
          </CardContent>
        </Card>
      )}

      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create Surrogate Model</DialogTitle>
        <DialogContent>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

          <TextField
            fullWidth
            label="Model Name"
            margin="normal"
            value={createForm.name}
            onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
            required
          />

          <TextField
            fullWidth
            label="Description"
            margin="normal"
            multiline
            rows={2}
            value={createForm.description}
            onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
          />

          <TextField
            fullWidth
            select
            label="Dataset"
            margin="normal"
            value={createForm.dataset_id}
            onChange={(e) => setCreateForm({ ...createForm, dataset_id: e.target.value })}
            required
          >
            {datasets.map((dataset) => (
              <MenuItem key={dataset.id} value={dataset.id}>
                {dataset.name}
              </MenuItem>
            ))}
          </TextField>

          <TextField
            fullWidth
            select
            label="Algorithm"
            margin="normal"
            value={createForm.algorithm}
            onChange={(e) => setCreateForm({ ...createForm, algorithm: e.target.value })}
            required
          >
            <MenuItem value="gaussian_process">Gaussian Process</MenuItem>
            <MenuItem value="neural_network">Neural Network</MenuItem>
            <MenuItem value="polynomial_chaos">Polynomial Chaos</MenuItem>
            <MenuItem value="random_forest">Random Forest</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button onClick={handleCreate} disabled={createLoading}>
            {createLoading ? 'Creating...' : 'Create Model'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}