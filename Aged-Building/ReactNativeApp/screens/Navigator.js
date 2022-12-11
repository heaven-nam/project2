import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import MainScreen from './MainScreen';
import CameraScreen from './CameraScreen';

const Stack = createStackNavigator();

export default function Navigator() {

    return (
        <NavigationContainer>
            <Stack.Navigator>

                <Stack.Screen
                    name="MAIN"
                    component={MainScreen}
                    options={{
                        title: '메인화면title',
                        headerShown: true // set a header if it expose or not
                    }} />

                <Stack.Screen
                    name="CAMERA"
                    component={CameraScreen}
                    options={{
                        title: '카메라',
                        headerShown: true
                    }} />

            </Stack.Navigator>
        </NavigationContainer>
    );
}