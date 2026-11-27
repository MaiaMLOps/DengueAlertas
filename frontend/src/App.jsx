import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import TestPerformance from './components/TestPerformance';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Layout />}>
                    <Route index element={<Dashboard />} />
                    <Route path="desempeno" element={<TestPerformance />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

export default App;