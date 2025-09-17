import React, { createContext, useContext, useReducer, ReactNode } from 'react';

// Define the UI state structure
export interface UIState {
  navigation: {
    isAnalysisRunning: boolean;
    zoomLevel: number;
    showGrid: boolean;
    showModelTree: boolean;
  };
  statusBar: {
    showSystemDetails: boolean;
  };
  visualizations: {
    showControls: boolean;
    stressScale: number;
    contourLevels: number;
    animationSpeed: number;
    showColorbar: boolean;
    meshOpacity: number;
    isAnimating: boolean;
    selectedVisualization: number;
  };
  modelTree: {
    expanded: Set<string>;
    selected: string[];
  };
  theme: {
    darkMode: boolean;
    primaryColor: string;
  };
}

// Define action types
export type UIAction =
  | { type: 'SET_ANALYSIS_RUNNING'; payload: boolean }
  | { type: 'SET_ZOOM_LEVEL'; payload: number }
  | { type: 'TOGGLE_GRID'; }
  | { type: 'TOGGLE_MODEL_TREE'; }
  | { type: 'TOGGLE_SYSTEM_DETAILS'; }
  | { type: 'TOGGLE_VIZ_CONTROLS'; }
  | { type: 'SET_STRESS_SCALE'; payload: number }
  | { type: 'SET_CONTOUR_LEVELS'; payload: number }
  | { type: 'SET_MESH_OPACITY'; payload: number }
  | { type: 'TOGGLE_COLORBAR'; }
  | { type: 'SET_ANIMATION_STATE'; payload: boolean }
  | { type: 'SET_SELECTED_VISUALIZATION'; payload: number }
  | { type: 'SET_MODEL_TREE_EXPANDED'; payload: Set<string> }
  | { type: 'SET_MODEL_TREE_SELECTED'; payload: string[] }
  | { type: 'RESET_VIEW'; }
  | { type: 'TOGGLE_DARK_MODE'; };

// Initial state
const initialState: UIState = {
  navigation: {
    isAnalysisRunning: false,
    zoomLevel: 1,
    showGrid: true,
    showModelTree: true,
  },
  statusBar: {
    showSystemDetails: false,
  },
  visualizations: {
    showControls: true,
    stressScale: 1,
    contourLevels: 10,
    animationSpeed: 1,
    showColorbar: true,
    meshOpacity: 0.7,
    isAnimating: false,
    selectedVisualization: 0,
  },
  modelTree: {
    expanded: new Set(['project', 'geometry', 'mesh']),
    selected: ['project'],
  },
  theme: {
    darkMode: true,
    primaryColor: '#00a8ff',
  },
};

// Reducer function
function uiReducer(state: UIState, action: UIAction): UIState {
  switch (action.type) {
    case 'SET_ANALYSIS_RUNNING':
      return {
        ...state,
        navigation: { ...state.navigation, isAnalysisRunning: action.payload }
      };

    case 'SET_ZOOM_LEVEL':
      return {
        ...state,
        navigation: { ...state.navigation, zoomLevel: action.payload }
      };

    case 'TOGGLE_GRID':
      return {
        ...state,
        navigation: { ...state.navigation, showGrid: !state.navigation.showGrid }
      };

    case 'TOGGLE_MODEL_TREE':
      return {
        ...state,
        navigation: { ...state.navigation, showModelTree: !state.navigation.showModelTree }
      };

    case 'TOGGLE_SYSTEM_DETAILS':
      return {
        ...state,
        statusBar: { ...state.statusBar, showSystemDetails: !state.statusBar.showSystemDetails }
      };

    case 'TOGGLE_VIZ_CONTROLS':
      return {
        ...state,
        visualizations: { ...state.visualizations, showControls: !state.visualizations.showControls }
      };

    case 'SET_STRESS_SCALE':
      return {
        ...state,
        visualizations: { ...state.visualizations, stressScale: action.payload }
      };

    case 'SET_CONTOUR_LEVELS':
      return {
        ...state,
        visualizations: { ...state.visualizations, contourLevels: action.payload }
      };

    case 'SET_MESH_OPACITY':
      return {
        ...state,
        visualizations: { ...state.visualizations, meshOpacity: action.payload }
      };

    case 'TOGGLE_COLORBAR':
      return {
        ...state,
        visualizations: { ...state.visualizations, showColorbar: !state.visualizations.showColorbar }
      };

    case 'SET_ANIMATION_STATE':
      return {
        ...state,
        visualizations: { ...state.visualizations, isAnimating: action.payload }
      };

    case 'SET_SELECTED_VISUALIZATION':
      return {
        ...state,
        visualizations: { ...state.visualizations, selectedVisualization: action.payload }
      };

    case 'SET_MODEL_TREE_EXPANDED':
      return {
        ...state,
        modelTree: { ...state.modelTree, expanded: action.payload }
      };

    case 'SET_MODEL_TREE_SELECTED':
      return {
        ...state,
        modelTree: { ...state.modelTree, selected: action.payload }
      };

    case 'RESET_VIEW':
      return {
        ...state,
        navigation: { ...state.navigation, zoomLevel: 1 },
        visualizations: {
          ...state.visualizations,
          stressScale: 1,
          contourLevels: 10,
          meshOpacity: 0.7,
          showColorbar: true
        }
      };

    case 'TOGGLE_DARK_MODE':
      return {
        ...state,
        theme: { ...state.theme, darkMode: !state.theme.darkMode }
      };

    default:
      return state;
  }
}

// Create context
interface UIContextType {
  state: UIState;
  dispatch: React.Dispatch<UIAction>;
}

const UIStateContext = createContext<UIContextType | undefined>(undefined);

// Provider component
export function UIStateProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(uiReducer, initialState);

  return (
    <UIStateContext.Provider value={{ state, dispatch }}>
      {children}
    </UIStateContext.Provider>
  );
}

// Custom hook to use the UI state
export function useUIState() {
  const context = useContext(UIStateContext);
  if (context === undefined) {
    throw new Error('useUIState must be used within a UIStateProvider');
  }
  return context;
}

// Utility functions for common actions
export const uiActions = {
  setAnalysisRunning: (running: boolean): UIAction => ({
    type: 'SET_ANALYSIS_RUNNING',
    payload: running
  }),

  setZoomLevel: (level: number): UIAction => ({
    type: 'SET_ZOOM_LEVEL',
    payload: level
  }),

  toggleGrid: (): UIAction => ({ type: 'TOGGLE_GRID' }),

  toggleModelTree: (): UIAction => ({ type: 'TOGGLE_MODEL_TREE' }),

  setStressScale: (scale: number): UIAction => ({
    type: 'SET_STRESS_SCALE',
    payload: scale
  }),

  resetView: (): UIAction => ({ type: 'RESET_VIEW' }),

  toggleAnimation: (isAnimating: boolean): UIAction => ({
    type: 'SET_ANIMATION_STATE',
    payload: isAnimating
  })
};