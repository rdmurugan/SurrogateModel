import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Alert,
  CircularProgress,
  Paper,
  Tabs,
  Tab,
  useTheme,
  alpha,
  Tooltip,
  IconButton,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  ExpandMore,
  Psychology,
  Timeline,
  AccountTree,
  AutoFixHigh,
  TrendingUp,
  Memory,
  Settings,
  Visibility,
  Download,
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  Analytics,
  ScatterPlot,
  ShowChart,
  BubbleChart,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import axios from 'axios';

interface NextGenCapabilities {
  bayesian_neural_networks: {
    description: string;
    features: string[];
    use_cases: string[];
  };
  graph_neural_networks: {
    description: string;
    features: string[];
    use_cases: string[];
  };
  transformer_models: {
    description: string;
    features: string[];
    use_cases: string[];
  };
}

interface NextGenSession {
  session_id: string;
  session_type: string;
  config: any;
  status: string;
  created_at: string;
  progress?: number;
  metrics?: {
    accuracy: number;
    loss: number;
    uncertainty: number;
    epoch: number;
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`nextgen-tabpanel-${index}`}
      aria-labelledby={`nextgen-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function NextGenML() {
  const theme = useTheme();
  const [capabilities, setCapabilities] = useState<NextGenCapabilities | null>(null);
  const [sessions, setSessions] = useState<NextGenSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedTab, setSelectedTab] = useState(0);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [selectedModel, setSelectedModel] = useState('bayesian');
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState(0.1);

  useEffect(() => {
    fetchCapabilities();
    fetchSessions();
  }, []);

  const fetchCapabilities = async () => {
    try {
      const response = await axios.get('/api/v1/nextgen-ml/capabilities');
      setCapabilities(response.data.next_generation_ml_capabilities);
    } catch (error) {
      console.error('Failed to fetch capabilities:', error);
      setError('Failed to load next-generation ML capabilities');
    }
  };

  const fetchSessions = async () => {
    try {
      setLoading(true);
      // Note: This would require authentication in production
      // const response = await axios.get('/api/v1/nextgen-ml/sessions');
      // setSessions(response.data.sessions || []);
      setSessions([]); // Mock empty for now
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const createDemoSession = async (sessionType: string) => {
    try {
      setError('');
      const demoConfigs = {
        bayesian: {
          input_dim: 3,
          output_dim: 1,
          hidden_layers: [64, 32],
          activation: 'relu',
          ensemble_size: 5,
          use_mc_dropout: true,
          heteroscedastic: true,
        },
        graph: {
          input_dim: 3,
          output_dim: 1,
          conv_type: 'gat',
          use_geometric_attention: true,
          task_type: 'node_prediction',
        },
        transformer: {
          input_dim: 5,
          output_dim: 2,
          d_model: 128,
          nhead: 8,
          transformer_type: 'optimization',
          use_feature_attention: true,
        },
      };

      const config = demoConfigs[sessionType as keyof typeof demoConfigs];

      // Note: This would require authentication in production
      // const response = await axios.post(`/api/v1/nextgen-ml/${sessionType}/sessions`, config);
      // console.log('Session created:', response.data);

      // Show demo message instead
      setError(`Demo: Would create ${sessionType} session with config: ${JSON.stringify(config, null, 2)}`);
    } catch (error) {
      console.error('Failed to create session:', error);
      setError('Failed to create demo session. Authentication required.');
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'bayesian_neural_networks':
        return <Psychology color="primary" />;
      case 'graph_neural_networks':
        return <AccountTree color="secondary" />;
      case 'transformer_models':
        return <Timeline color="success" />;
      default:
        return <AutoFixHigh />;
    }
  };

  const getFeatureIcon = (feature: string) => {
    if (feature.toLowerCase().includes('uncertainty') || feature.toLowerCase().includes('inference')) {
      return <TrendingUp color="primary" />;
    }
    if (feature.toLowerCase().includes('attention') || feature.toLowerCase().includes('geometric')) {
      return <Memory color="secondary" />;
    }
    return <AutoFixHigh color="action" />;
  };

  if (loading && !capabilities) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{
      flexGrow: 1,
      p: 3,
      background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.secondary.light, 0.05)} 100%)`,
      minHeight: '100vh'
    }}>
      {/* Header Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" sx={{
          fontWeight: 700,
          color: theme.palette.primary.main,
          mb: 1,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}>
          <BubbleChart sx={{ fontSize: 40 }} />
          Next-Generation ML Studio
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Advanced Machine Learning with Uncertainty Quantification & Neural Architecture Design
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            icon={<TrendingUp />}
            label="Bayesian Inference"
            color="primary"
            variant="outlined"
          />
          <Chip
            icon={<Timeline />}
            label="Graph Neural Networks"
            color="secondary"
            variant="outlined"
          />
          <Chip
            icon={<Memory />}
            label="Transformer Attention"
            color="success"
            variant="outlined"
          />
        </Box>
      </Box>

      {error && (
        <Alert severity="info" sx={{ mb: 3, bgcolor: alpha(theme.palette.info.light, 0.1) }}>
          {error}
        </Alert>
      )}

      {/* Model Capabilities Overview */}
      {capabilities && (
        <Paper elevation={3} sx={{ mb: 4, p: 3, bgcolor: alpha(theme.palette.background.paper, 0.95) }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Analytics />
            Model Architecture Capabilities
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(capabilities).map(([key, capability]) => (
              <Grid item xs={12} md={4} key={key}>
                <Card
                  elevation={2}
                  sx={{
                    height: '100%',
                    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.05)} 0%, ${alpha(theme.palette.secondary.light, 0.02)} 100%)`,
                    border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      elevation: 4,
                      transform: 'translateY(-2px)',
                    }
                  }}
                >
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      {getIcon(key)}
                      <Typography variant="h6" component="h3" sx={{ ml: 1, textTransform: 'capitalize', fontWeight: 600 }}>
                        {key.replace(/_/g, ' ')}
                      </Typography>
                    </Box>

                    <Typography variant="body2" color="text.secondary" paragraph sx={{ minHeight: 60 }}>
                      {capability.description}
                    </Typography>

                    <Accordion elevation={0} sx={{ bgcolor: 'transparent' }}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          Technical Features ({capability.features.length})
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <List dense>
                          {capability.features.map((feature: string, index: number) => (
                            <ListItem key={index} disablePadding>
                              <ListItemIcon sx={{ minWidth: 32 }}>
                                {getFeatureIcon(feature)}
                              </ListItemIcon>
                              <ListItemText
                                primary={feature}
                                primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </AccordionDetails>
                    </Accordion>

                    <Accordion elevation={0} sx={{ bgcolor: 'transparent' }}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          Engineering Applications
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Box>
                          {capability.use_cases.map((useCase: string, index: number) => (
                            <Chip
                              key={index}
                              label={useCase}
                              size="small"
                              variant="outlined"
                              sx={{
                                mr: 1,
                                mb: 1,
                                bgcolor: alpha(theme.palette.primary.main, 0.05),
                                border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
                              }}
                            />
                          ))}
                        </Box>
                      </AccordionDetails>
                    </Accordion>

                    <Box mt={2}>
                      <Button
                        variant="contained"
                        size="medium"
                        onClick={() => createDemoSession(key.split('_')[0])}
                        startIcon={<AutoFixHigh />}
                        fullWidth
                        sx={{
                          background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
                          fontWeight: 600,
                          '&:hover': {
                            background: `linear-gradient(45deg, ${theme.palette.primary.dark} 30%, ${theme.palette.secondary.dark} 90%)`,
                          }
                        }}
                      >
                        Launch Interactive Session
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {/* Active Sessions */}
      <Paper elevation={3} sx={{ mb: 4, bgcolor: alpha(theme.palette.background.paper, 0.95) }}>
        <Box sx={{ p: 3, borderBottom: `1px solid ${theme.palette.divider}` }}>
          <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline />
            Active Model Sessions
            {loading && <CircularProgress size={20} sx={{ ml: 2 }} />}
          </Typography>
        </Box>
        <Box sx={{ p: 3 }}>
          {sessions.length === 0 ? (
            <Alert severity="info" sx={{ bgcolor: alpha(theme.palette.info.light, 0.1) }}>
              <Typography variant="body1">
                🚀 Ready to start? Create your first next-generation ML session using the buttons above!
              </Typography>
            </Alert>
          ) : (
            <Grid container spacing={3}>
              {sessions.map((session) => (
                <Grid item xs={12} md={6} lg={4} key={session.session_id}>
                  <Card
                    elevation={2}
                    sx={{
                      bgcolor: alpha(theme.palette.background.paper, 0.8),
                      border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                      '&:hover': { elevation: 4 }
                    }}
                  >
                    <CardContent>
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        {session.session_type} Session
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        ID: {session.session_id.substring(0, 8)}...
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Chip
                          label={session.status}
                          size="small"
                          color={session.status === 'active' ? 'success' : 'info'}
                          sx={{ fontWeight: 600 }}
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        Created: {new Date(session.created_at).toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Box>
      </Paper>

      {/* Integration Status */}
      <Paper elevation={3} sx={{ p: 3, bgcolor: alpha(theme.palette.background.paper, 0.95) }}>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShowChart />
          Platform Integration Status
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ height: '100%', bgcolor: alpha(theme.palette.success.light, 0.05) }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="success.main" sx={{ fontWeight: 600 }}>
                  ✅ Production Ready APIs
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemIcon><PlayArrow color="success" /></ListItemIcon>
                    <ListItemText
                      primary="Session Management"
                      secondary="Create, train, and deploy advanced ML models"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><TrendingUp color="success" /></ListItemIcon>
                    <ListItemText
                      primary="Uncertainty Quantification"
                      secondary="Bayesian inference and confidence calibration"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Memory color="success" /></ListItemIcon>
                    <ListItemText
                      primary="Attention Analysis"
                      secondary="Feature importance and model interpretability"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ height: '100%', bgcolor: alpha(theme.palette.secondary.light, 0.05) }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="secondary.main" sx={{ fontWeight: 600 }}>
                  🚀 Advanced Capabilities
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemIcon><BubbleChart color="secondary" /></ListItemIcon>
                    <ListItemText
                      primary="Multi-Modal Fusion"
                      secondary="Combine simulation, experimental, and CAD data"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><ScatterPlot color="secondary" /></ListItemIcon>
                    <ListItemText
                      primary="Transfer Learning"
                      secondary="Adapt pre-trained models to new domains"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Timeline color="secondary" /></ListItemIcon>
                    <ListItemText
                      primary="Geometric Deep Learning"
                      secondary="Process complex mesh and CAD geometries"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}