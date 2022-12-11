import React, { useEffect, useState } from 'react';
import { 
  TouchableOpacity, 
  StyleSheet, 
  View, Button, 
  Text, 
  SafeAreaView,
  Image, 
  ScrollView, 
  StatusBar } from 'react-native';
import { NavigationContainer, StackRouter } from '@react-navigation/native';
import { createStackNavigator, CardStyleInterpolators } from '@react-navigation/stack';
import {RNCamera} from 'react-native-camera';
//import {StatusBar} from 'expo-status-bar';
import {Camera} from 'expo-camera'; // caemra 사용 가능하게 해주는 API
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import MainScreen from './screens/MainScreen';
//import Album from './screens/Album';

const CameraScreen = ({navigation}) => {
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();

  if (!permission) {
    return <View></View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{textAlign:'center'}}>카메라 권한을 주셔야 촬영이 가능합니다.</Text>
        <Button onPress={requestPermission} title="권한 승인"></Button>
      </View>
    );
  }

  const toggleCameraType = () => {
    setType(current => (current===Camera.Constants.Type.back ? Camera.Constants.Type.front : Camera.Constants.Type.back));
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type}>
        <View style={styles.cameraButtonContainer}>
          <TouchableOpacity style={styles.camerascreen_button} onPress={toggleCameraType}>
            <Text style={styles.camerascreen_text}>
              화면 전환
            </Text>
          </TouchableOpacity>
        </View>
      </Camera>
    </View>
  );
}

const Album = ({navigation}) => {
  return (
    <View style={styles.container}>
      <Text>Album</Text>
      <Button title="Home으로 돌아가기" onPress={() => navigation.pop()} />
    </View>
  );
}
// navigation
// .goback 한칸 전으로 이동
// .pop 한칸 전으로 이동
// .popToTop 처음으로 이동

const Stack = createStackNavigator();

function MyStack(){
  return (
    <Stack.Navigator>
      <Stack.Screen 
        name="Home" 
        component={MainScreen}
        options={styles.header}
        >
      </Stack.Screen>

      <Stack.Screen 
        name="Camera" 
        component={CameraScreen}
        options={styles.header}
        >
      </Stack.Screen>
      
    {/* 
      <Stack.Screen 
        name="Camera" 
        component={CameraButton}
        options={styles.album}
        >
      </Stack.Screen> */}

      <Stack.Screen 
        name="Album" 
        component={Album}
        options={styles.header}
        >
      </Stack.Screen>
    </Stack.Navigator>
  );
}

export default function App () {
  
  return (
    <NavigationContainer>
      <MyStack></MyStack>
    </NavigationContainer>
  );
}


const styles = StyleSheet.create({
  header:{
    headerTintiColor: '#fff',
    headerStyle: {backgroundColor:'tomato'},
    headerTitleStyle: {
      fontWeight: 'bold',
      fontSize: 20,
    },
  },
  album:{
    headerTintiColor: '#fff',
    headerStyle: {
      backgroundColor:'tomato'
    },
    headerTitleStyle: {
      fontWeight: 'bold',
      fontSize: 20,
    },
    CardStyleInterpolators: CardStyleInterpolators.forRevealFromBottomAndroid
  },
  container: {
    flex: 1,
    alignItems:'center',
    justifyContent:'center',
    //marginHorizontal:16,
    //padding: 20,
    //margin: 10,
    backgroundColor: '#fffacd'
  },
  button_top: {
    flex: 0.07,
    width: 200,
    //height: 10,
    backgroundColor:"#008000",
    borderWidth: 2,
    borderRadius: 20,
    //borderTopLeftRadius: 20,
    //borderTopRightRadius: 20,
    //borderBottomLeftRadius: 20,
    //borderBottomRightRadius: 20,
    padding: 0,
    margin: 50,
  },
  button_bottom: {
    flex: 0.07,
    width: 200,
    //height: 10,
    backgroundColor:"#008000",
    borderWidth: 2,
    borderRadius: 20,
    //borderTopLeftRadius: 20,
    //borderTopRightRadius: 20,
    //borderBottomLeftRadius: 20,
    //borderBottomRightRadius: 20,
    padding: 0,
    margin: 50,
  },
  text: {
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'center'
  },
  touchableopacity: {
    width: 200,
    borderRadius: 20,
    backgroundColor: "#008000",//'#14274e',
    borderWidth: 2,
    //flexDirection: 'center',
    justifyContent: 'center',
    alignItems: 'center',
    height: 42,
  },
  camera:{
    flex:1,
  },
  cameraButtonContainer:{
    flex:1,
    flexDirection:'row',
    backgroundColor:'transparent',
    margin:64,
  },
  camerascreen_button: {
    width: 130,
    borderRadius: 4,
    backgroundColor: '#14274e',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    height: 40
  },
  camerascreen_text: {
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'center'
  },
});