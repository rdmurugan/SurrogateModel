import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  MenuItem,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from '@mui/material';
import { ExpandMore, PlayArrow } from '@mui/icons-material';
import axios from 'axios';

interface Model {
  id: number;
  name: string;
  algorithm: string;
  is_deployed: boolean;
}

interface PredictionResult {
  prediction_id: number;
  input_data: any;
  output_data: any;
  uncertainty_data: any;
  prediction_time_ms: number;
}

export default function Predictions() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelDetails, setModelDetails] = useState<any>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      fetchModelDetails(parseInt(selectedModel));
    }
  }, [selectedModel]);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/v1/models/');
      const deployedModels = response.data.filter((model: Model) => model.is_deployed);
      setModels(deployedModels);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchModelDetails = async (modelId: number) => {
    try {
      const response = await axios.get(`/api/v1/models/${modelId}`);
      setModelDetails(response.data);

      // Initialize input values
      const inputs = response.data.dataset.input_columns || [];
      const initialValues: Record<string, string> = {};
      inputs.forEach((col: string) => {
        initialValues[col] = '';
      });
      setInputValues(initialValues);
    } catch (error) {
      console.error('Error fetching model details:', error);
    }
  };

  const handlePredict = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    // Validate all inputs are filled
    const missingInputs = Object.entries(inputValues).filter(([_, value]) => !value);
    if (missingInputs.length > 0) {
      setError('Please fill in all input parameters');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Convert string values to numbers
      const numericInputs: Record<string, number> = {};
      Object.entries(inputValues).forEach(([key, value]) => {
        numericInputs[key] = parseFloat(value);
      });

      const response = await axios.post(`/api/v1/predictions/${selectedModel}/predict`, numericInputs);
      setPrediction(response.data);
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Predictions
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Make fast predictions using your trained surrogate models
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Input Parameters
              </Typography>

              {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

              <TextField
                fullWidth
                select
                label="Select Model"
                margin="normal"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {models.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    {model.name} ({model.algorithm})
                  </MenuItem>
                ))}
              </TextField>

              {modelDetails && modelDetails.dataset.input_columns && (
                <Box sx={{ mt: 2 }}>
                  {modelDetails.dataset.input_columns.map((column: string) => (
                    <TextField
                      key={column}
                      fullWidth
                      label={column}
                      margin="normal"
                      type="number"
                      value={inputValues[column] || ''}
                      onChange={(e) => setInputValues({
                        ...inputValues,
                        [column]: e.target.value
                      })}
                    />
                  ))}

                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<PlayArrow />}
                    onClick={handlePredict}
                    disabled={loading || !selectedModel}
                    sx={{ mt: 2 }}
                  >
                    {loading ? 'Predicting...' : 'Make Prediction'}
                  </Button>
                </Box>
              )}

              {models.length === 0 && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  No deployed models available. Create and train a model first.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {prediction && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Prediction Results
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={`${prediction.prediction_time_ms.toFixed(2)} ms`}
                    color="success"
                    size="small"
                  />
                </Box>

                <Typography variant="subtitle2" gutterBottom>
                  Predicted Values:
                </Typography>
                {Object.entries(prediction.output_data).map(([key, value]) => (
                  <Typography key={key} variant="body2" sx={{ mb: 1 }}>
                    <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : value}
                  </Typography>
                ))}

                {prediction.uncertainty_data && (
                  <Accordion sx={{ mt: 2 }}>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="subtitle2">
                        Uncertainty Quantification
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {Object.entries(prediction.uncertainty_data).map(([key, uncertainty]: [string, any]) => (
                        <Box key={key} sx={{ mb: 2 }}>
                          <Typography variant="body2" fontWeight="bold">
                            {key}:
                          </Typography>
                          <Typography variant="body2">
                            95% CI: [{uncertainty.confidence_interval_95[0].toFixed(4)}, {uncertainty.confidence_interval_95[1].toFixed(4)}]
                          </Typography>
                          <Typography variant="body2">
                            Std Dev: ±{uncertainty.standard_deviation.toFixed(4)}
                          </Typography>
                        </Box>
                      ))}
                    </AccordionDetails>
                  </Accordion>
                )}
              </CardContent>
            </Card>
          )}

          {!prediction && selectedModel && (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <PlayArrow sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Ready to Predict
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Fill in the input parameters and click "Make Prediction"
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}