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
  LinearProgress,
  Alert,
} from '@mui/material';
import { Add, Upload } from '@mui/icons-material';
import axios from 'axios';

interface Dataset {
  id: number;
  name: string;
  description: string;
  status: string;
  num_samples: number;
  input_columns: string[];
  output_columns: string[];
  created_at: string;
}

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadForm, setUploadForm] = useState({
    name: '',
    description: '',
    file: null as File | null,
    inputColumns: '',
    outputColumns: '',
  });
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get('/api/v1/datasets/');
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!uploadForm.file || !uploadForm.name || !uploadForm.inputColumns || !uploadForm.outputColumns) {
      setError('Please fill in all required fields');
      return;
    }

    setUploadLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', uploadForm.file);
      formData.append('name', uploadForm.name);
      formData.append('description', uploadForm.description);
      formData.append('input_columns', uploadForm.inputColumns);
      formData.append('output_columns', uploadForm.outputColumns);

      await axios.post('/api/v1/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setUploadOpen(false);
      setUploadForm({
        name: '',
        description: '',
        file: null,
        inputColumns: '',
        outputColumns: '',
      });
      fetchDatasets();
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploadLoading(false);
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Datasets</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setUploadOpen(true)}
        >
          Upload Dataset
        </Button>
      </Box>

      <Grid container spacing={3}>
        {datasets.map((dataset) => (
          <Grid item xs={12} md={6} key={dataset.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {dataset.name}
                </Typography>
                {dataset.description && (
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {dataset.description}
                  </Typography>
                )}
                <Box display="flex" gap={1} mb={2}>
                  <Chip
                    label={dataset.status}
                    color={dataset.status === 'processed' ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip label={`${dataset.num_samples} samples`} size="small" />
                </Box>
                <Typography variant="body2" gutterBottom>
                  <strong>Inputs:</strong> {dataset.input_columns.join(', ')}
                </Typography>
                <Typography variant="body2">
                  <strong>Outputs:</strong> {dataset.output_columns.join(', ')}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Created: {new Date(dataset.created_at).toLocaleDateString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {datasets.length === 0 && (
        <Card sx={{ mt: 3 }}>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Upload sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No datasets uploaded yet
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Upload your simulation data to get started with surrogate modeling
            </Typography>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setUploadOpen(true)}
              sx={{ mt: 2 }}
            >
              Upload Your First Dataset
            </Button>
          </CardContent>
        </Card>
      )}

      <Dialog open={uploadOpen} onClose={() => setUploadOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Upload Dataset</DialogTitle>
        <DialogContent>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

          <TextField
            fullWidth
            label="Dataset Name"
            margin="normal"
            value={uploadForm.name}
            onChange={(e) => setUploadForm({ ...uploadForm, name: e.target.value })}
            required
          />

          <TextField
            fullWidth
            label="Description"
            margin="normal"
            multiline
            rows={2}
            value={uploadForm.description}
            onChange={(e) => setUploadForm({ ...uploadForm, description: e.target.value })}
          />

          <Button
            variant="outlined"
            component="label"
            fullWidth
            sx={{ mt: 2, mb: 2 }}
          >
            {uploadForm.file ? uploadForm.file.name : 'Choose File (CSV or Excel)'}
            <input
              type="file"
              hidden
              accept=".csv,.xlsx,.xls"
              onChange={(e) => setUploadForm({ ...uploadForm, file: e.target.files?.[0] || null })}
            />
          </Button>

          <TextField
            fullWidth
            label="Input Columns (JSON array)"
            margin="normal"
            placeholder='["length", "width", "thickness"]'
            value={uploadForm.inputColumns}
            onChange={(e) => setUploadForm({ ...uploadForm, inputColumns: e.target.value })}
            required
          />

          <TextField
            fullWidth
            label="Output Columns (JSON array)"
            margin="normal"
            placeholder='["stress", "displacement"]'
            value={uploadForm.outputColumns}
            onChange={(e) => setUploadForm({ ...uploadForm, outputColumns: e.target.value })}
            required
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadOpen(false)}>Cancel</Button>
          <Button onClick={handleUpload} disabled={uploadLoading}>
            {uploadLoading ? 'Uploading...' : 'Upload'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}