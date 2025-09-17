import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  Chip,
  LinearProgress,
  Avatar,
  IconButton,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  useTheme,
  alpha,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  Analytics,
  Speed,
  Engineering,
  Memory,
  Timeline,
  PlayArrow,
  Pause,
  Stop,
  Settings,
  Refresh,
  Assessment,
  BubbleChart,
  ShowChart,
  ScatterPlot,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import axios from 'axios';

interface MetricCard {
  title: string;
  value: string;
  change: string;
  icon: React.ReactNode;
  color: string;
}

interface Session {
  id: string;
  type: string;
  status: string;
  accuracy: number;
  uncertainty: number;
  created: string;
}

export default function Dashboard() {
  const theme = useTheme();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSessionData();
    const interval = setInterval(fetchSessionData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchSessionData = async () => {
    try {
      setLoading(true);
      setSessions([
        {
          id: '31165cad-3684-4ad3-bf2b-322612b07098',
          type: 'Bayesian NN',
          status: 'Training',
          accuracy: 94.2,
          uncertainty: 0.023,
          created: '2024-01-15 14:30'
        },
        {
          id: 'd0e35486-ee4f-40db-95c2-09d1687a6df6',
          type: 'Graph NN',
          status: 'Completed',
          accuracy: 97.8,
          uncertainty: 0.012,
          created: '2024-01-15 13:45'
        },
        {
          id: 'bc008749-e678-44b5-b041-49b79bd4ba01',
          type: 'Transformer',
          status: 'Ready',
          accuracy: 92.1,
          uncertainty: 0.034,
          created: '2024-01-15 12:20'
        }
      ]);
    } catch (error) {
      console.error('Failed to fetch session data:', error);
    } finally {
      setLoading(false);
    }
  };

  const metricCards: MetricCard[] = [
    {
      title: 'Active Models',
      value: '12',
      change: '+3 today',
      icon: <Engineering sx={{ fontSize: 30 }} />,
      color: theme.palette.primary.main,
    },
    {
      title: 'Accuracy Rate',
      value: '94.7%',
      change: '+2.1% this week',
      icon: <TrendingUp sx={{ fontSize: 30 }} />,
      color: theme.palette.success.main,
    },
    {
      title: 'Predictions/Hour',
      value: '2,847',
      change: '+15.3% vs last hour',
      icon: <Speed sx={{ fontSize: 30 }} />,
      color: theme.palette.info.main,
    },
    {
      title: 'System Load',
      value: '67%',
      change: 'Normal range',
      icon: <Memory sx={{ fontSize: 30 }} />,
      color: theme.palette.warning.main,
    },
  ];

  const performanceData = {
    data: [
      {
        x: ['Bayesian NN', 'Graph NN', 'Transformer', 'Classical ML'],
        y: [94.2, 97.8, 92.1, 87.3],
        type: 'bar' as const,
        marker: {
          color: ['#1976d2', '#2e7d32', '#ed6c02', '#d32f2f'],
          line: { color: 'rgba(0,0,0,0.1)', width: 1 }
        },
        name: 'Accuracy (%)',
        text: ['94.2%', '97.8%', '92.1%', '87.3%'],
        textposition: 'auto' as const,
      }
    ],
    layout: {
      title: {
        text: 'Model Performance Comparison',
        font: { size: 16, family: 'Roboto, sans-serif' }
      },
      xaxis: { title: 'Model Type' },
      yaxis: { title: 'Accuracy (%)', range: [80, 100] },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Roboto, sans-serif' },
      margin: { t: 50, r: 20, b: 50, l: 50 }
    },
    config: { responsive: true, displayModeBar: false }
  };

  const uncertaintyData = {
    data: [
      {
        x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        y: [0.023, 0.019, 0.025, 0.031, 0.018, 0.022, 0.027, 0.020, 0.024, 0.021],
        error_y: {
          type: 'data' as const,
          array: [0.003, 0.002, 0.004, 0.003, 0.002, 0.003, 0.004, 0.002, 0.003, 0.002],
          visible: true,
          color: '#1976d2'
        },
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        marker: { color: '#1976d2', size: 8 },
        line: { color: '#1976d2', width: 2 },
        name: 'Uncertainty',
      }
    ],
    layout: {
      title: {
        text: 'Uncertainty Evolution During Training',
        font: { size: 16, family: 'Roboto, sans-serif' }
      },
      xaxis: { title: 'Training Epoch' },
      yaxis: { title: 'Uncertainty (σ)' },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Roboto, sans-serif' },
      margin: { t: 50, r: 20, b: 50, l: 50 }
    },
    config: { responsive: true, displayModeBar: false }
  };

  const systemMetricsData = {
    data: [
      {
        x: ['CPU', 'Memory', 'GPU', 'Storage'],
        y: [67, 45, 82, 34],
        type: 'bar' as const,
        marker: {
          color: ['#f44336', '#ff9800', '#4caf50', '#2196f3'],
          line: { color: 'rgba(0,0,0,0.1)', width: 1 }
        },
        name: 'Usage (%)',
      }
    ],
    layout: {
      title: {
        text: 'System Resource Utilization',
        font: { size: 16, family: 'Roboto, sans-serif' }
      },
      xaxis: { title: 'Resource Type' },
      yaxis: { title: 'Usage (%)', range: [0, 100] },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Roboto, sans-serif' },
      margin: { t: 50, r: 20, b: 50, l: 50 }
    },
    config: { responsive: true, displayModeBar: false }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Training': return theme.palette.warning.main;
      case 'Completed': return theme.palette.success.main;
      case 'Ready': return theme.palette.info.main;
      case 'Failed': return theme.palette.error.main;
      default: return theme.palette.grey[500];
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Training': return <Refresh className="rotating" />;
      case 'Completed': return <Assessment />;
      case 'Ready': return <PlayArrow />;
      default: return <Pause />;
    }
  };

  return (
    <Box sx={{
      flexGrow: 1,
      p: 3,
      background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.secondary.light, 0.05)} 100%)`,
      minHeight: '100vh'
    }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" sx={{
          fontWeight: 700,
          color: theme.palette.primary.main,
          mb: 1,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}>
          <Analytics sx={{ fontSize: 40 }} />
          Engineering Analytics Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Advanced Surrogate Modeling & Machine Learning Platform
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            icon={<TrendingUp />}
            label="Real-time Monitoring"
            color="success"
            variant="outlined"
          />
          <Chip
            icon={<Timeline />}
            label="Uncertainty Quantification"
            color="primary"
            variant="outlined"
          />
          <Chip
            icon={<BubbleChart />}
            label="Graph Neural Networks"
            color="secondary"
            variant="outlined"
          />
        </Box>
      </Box>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metricCards.map((metric, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              elevation={3}
              sx={{
                height: '100%',
                background: `linear-gradient(135deg, ${alpha(metric.color, 0.1)} 0%, ${alpha(metric.color, 0.05)} 100%)`,
                border: `1px solid ${alpha(metric.color, 0.2)}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  elevation: 6,
                  transform: 'translateY(-2px)',
                }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Avatar sx={{ bgcolor: metric.color, width: 56, height: 56 }}>
                    {metric.icon}
                  </Avatar>
                  <IconButton size="small" sx={{ color: metric.color }}>
                    <Refresh />
                  </IconButton>
                </Box>
                <Typography color="text.secondary" gutterBottom variant="h6">
                  {metric.title}
                </Typography>
                <Typography variant="h4" component="div" sx={{ fontWeight: 700, mb: 1 }}>
                  {metric.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {metric.change}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              backgroundColor: '#2d2d30',
              border: '1px solid #3e3e42',
              borderRadius: '2px',
              p: 1,
              height: 400
            }}
          >
            <Plot
              data={performanceData.data}
              layout={performanceData.layout}
              config={performanceData.config}
              style={{ width: '100%', height: '100%' }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              backgroundColor: '#2d2d30',
              border: '1px solid #3e3e42',
              borderRadius: '2px',
              p: 1,
              height: 400
            }}
          >
            <Plot
              data={uncertaintyData.data}
              layout={uncertaintyData.layout}
              config={uncertaintyData.config}
              style={{ width: '100%', height: '100%' }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            elevation={0}
            sx={{
              backgroundColor: '#2d2d30',
              border: '1px solid #3e3e42',
              borderRadius: '2px',
              p: 1,
              height: 400
            }}
          >
            <Plot
              data={systemMetricsData.data}
              layout={systemMetricsData.layout}
              config={systemMetricsData.config}
              style={{ width: '100%', height: '100%' }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            elevation={3}
            sx={{
              height: 400,
              background: alpha(theme.palette.background.paper, 0.9),
              backdropFilter: 'blur(10px)'
            }}
          >
            <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Timeline />
                Active Model Sessions
                {loading && <LinearProgress sx={{ width: 100, ml: 2 }} />}
              </Typography>
            </Box>
            <TableContainer sx={{ maxHeight: 320 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Model Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Accuracy</TableCell>
                    <TableCell align="right">Uncertainty</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sessions.map((session) => (
                    <TableRow key={session.id} hover>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Avatar sx={{ width: 24, height: 24, fontSize: 12 }}>
                            {session.type.charAt(0)}
                          </Avatar>
                          {session.type}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getStatusIcon(session.status)}
                          label={session.status}
                          size="small"
                          sx={{
                            bgcolor: alpha(getStatusColor(session.status), 0.1),
                            color: getStatusColor(session.status),
                            border: `1px solid ${getStatusColor(session.status)}`
                          }}
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {session.accuracy}%
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="text.secondary">
                          ±{session.uncertainty}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Tooltip title="Model Settings">
                          <IconButton size="small">
                            <Settings />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      <style>
        {`
          @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .rotating {
            animation: rotate 2s linear infinite;
          }
        `}
      </style>
    </Box>
  );
}